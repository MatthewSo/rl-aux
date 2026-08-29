"""The WAMAL-LM training loop.

Ported from `wamal/train_network.py`, keeping the bilevel structure (inner SGD
step on theta with `create_graph=True`, outer step on phi against the
post-update primary loss) and changing four things deliberately:

1. Arm behaviour is derived from `cfg.arm` instead of loose booleans, so an
   inconsistent configuration cannot be constructed.
2. Accuracy divides by the true batch length. The original divides by the
   configured `batch_size`, which under-reports whenever the last batch is short.
3. Model selection is on a validation split, not on test.
4. Only theta parameters take part in the inner step; the frozen backbone and
   phi's adapter are left out of the override dict.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from llm.losses import model_entropy, model_fit, scale_weights
from llm.paths import mark_complete


def inner_sgd_update(model, loss: torch.Tensor, lr: float,
                     first_order: bool = False) -> OrderedDict:
    """One differentiable SGD step on theta only -> `theta+`.

    `create_graph=True` is what makes the result a function of phi, which is the
    entire mechanism by which the label network receives a gradient.
    """
    names = model.theta_parameter_names()
    lookup = dict(model.named_parameters())
    params = [lookup[n] for n in names]

    grads = torch.autograd.grad(
        loss, params, create_graph=not first_order, allow_unused=True
    )

    fast = OrderedDict()
    for name, param, grad in zip(names, params, grads):
        fast[name] = param if grad is None else param - lr * grad
    return fast


def _auxiliary_targets(cfg, model, batch, device, detach: bool):
    """-> (soft target over C*psi, per-sample weight or None).

    `human` builds a one-hot over the true fine labels. That works without
    remapping only because the datasets guarantee `fine // psi == coarse`, which
    puts each fine label inside its own class's mask window.
    """
    if cfg.uses_human_labels:
        onehot = F.one_hot(batch["fine"], num_classes=cfg.num_auxiliary).float()
        return onehot, None

    sub_labels, raw_weights = model.forward_label(
        batch["input_ids"], batch["attention_mask"], batch["coarse"]
    )
    if detach:
        # Phase A only steps theta, so the phi backward is wasted work.
        sub_labels = sub_labels.detach()
        raw_weights = raw_weights.detach()
    return sub_labels, raw_weights


def _combine(cfg, primary_loss, auxiliary_loss, raw_weights):
    if not cfg.uses_auxiliary:
        return primary_loss.mean(), torch.zeros((), device=primary_loss.device)

    if cfg.uses_learned_weights and raw_weights is not None:
        factors = scale_weights(raw_weights.squeeze(-1), cfg.val_range)
        if cfg.normalize_batch_weights:
            factors = factors / factors.mean()
        aux = torch.mean(auxiliary_loss * factors)
    else:
        aux = torch.mean(auxiliary_loss)

    return primary_loss.mean() + aux, aux


def _cosine_similarity(model, primary_loss, auxiliary_loss) -> float:
    """Alignment between primary and auxiliary gradients on the shared adapters.

    Restricted to LoRA parameters. The original slices off "the last 8" entries
    of the gradient list to skip task-specific heads, which is a VGG-shaped
    assumption; naming the shared parameters explicitly is equivalent and does
    not silently mean something different on another architecture.
    """
    shared = [p for n, p in model.theta_named_parameters() if ".adapters." in n]
    if not shared:
        return 0.0
    g1 = torch.autograd.grad(primary_loss.mean(), shared, retain_graph=True,
                             allow_unused=True)
    g2 = torch.autograd.grad(auxiliary_loss.mean(), shared, retain_graph=True,
                             allow_unused=True)
    total, count = 0.0, 0
    for a, b in zip(g1, g2):
        if a is None or b is None:
            continue
        total += F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def evaluate(cfg, model, loader, device) -> dict:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    aux_correct = 0

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        primary, auxiliary = model(batch["input_ids"], batch["attention_mask"],
                                   mode="primary")
        loss = model_fit(primary, batch["coarse"], device, pri=True,
                         num_output=cfg.num_primary)

        n = batch["coarse"].size(0)
        loss_sum += loss.sum().item()
        correct += primary.argmax(dim=1).eq(batch["coarse"]).sum().item()
        # Agreement between the auxiliary head and the human fine labels. Only
        # meaningful where those labels exist and are anchored: see
        # llm/analysis/recovery.py, which shows this reads 1/psi even on a perfect
        # recovery because generated clusters are identified only up to a
        # permutation. On the invented-hierarchy tasks `fine` is a placeholder, so
        # scoring against it would be worse than useless.
        if cfg.has_fine_hierarchy:
            aux_correct += auxiliary.argmax(dim=1).eq(batch["fine"]).sum().item()
        total += n

    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "aux_vs_fine_acc": (aux_correct / max(total, 1)
                            if cfg.has_fine_hierarchy else None),
        "n": total,
    }


@torch.no_grad()
def test_assignment(cfg, model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """phi's sub-label assignment over the test split -> (sublabels, primary).

    Where there is no ground truth below the primary class, agreement between
    *independent seeds* is the available evidence that phi found real structure
    rather than fitting noise: two runs that partition the same examples the same
    way found something in the data, and two that disagree did not.

    The test split is the only place that comparison is possible, since the k-shot
    training subset differs by seed. Read it with
    `llm/analysis/recovery.py --inter-seed`.
    """
    model.eval()
    size = len(loader.dataset)
    sub_out = np.full(size, -1, dtype=np.int64)
    primary_out = np.full(size, -1, dtype=np.int64)

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        sub_labels, _ = model.forward_label(
            batch["input_ids"], batch["attention_mask"], batch["coarse"]
        )
        index = batch["index"].cpu().numpy()
        sub_out[index] = sub_labels.argmax(dim=1).cpu().numpy()
        primary_out[index] = batch["coarse"].cpu().numpy()

    return sub_out, primary_out


def train_wamal_text(cfg, device, model, loaders, theta_optimizer, theta_scheduler,
                     phi_optimizer, phi_scheduler, save_path, logger,
                     tracker=None) -> dict:
    save_path = Path(save_path)
    if tracker is None:
        from llm.tracking import NullTracker

        tracker = NullTracker()
    train_loader = loaders["train"]
    train_size = loaders["train_size"]

    best_val_acc = -1.0
    best_epoch = -1
    best_test_acc = float("nan")
    started = time.time()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dump_epochs = _dump_schedule(cfg)
    logger.print(f"training {cfg.cell_id()} | {cfg.total_epoch} epochs "
                 f"| {train_size} train examples | dumps at {sorted(dump_epochs)}")

    for epoch in range(cfg.total_epoch):
        # Mirrors the original's manual halving of the inner-loop LR, expressed
        # as a function of the epoch instead of mutating a captured variable.
        inner_lr = cfg.inner_lr * (cfg.gamma ** (epoch // cfg.step_size))

        train_stats = _train_epoch(
            cfg, model, train_loader, device, theta_optimizer, inner_lr,
            collect_dumps=epoch in dump_epochs, train_size=train_size,
        )

        meta_stats = {}
        if cfg.needs_meta_step:
            meta_stats = _meta_epoch(
                cfg, model, train_loader, device, phi_optimizer, inner_lr,
                query_loader=loaders.get("meta"),
            )

        val = evaluate(cfg, model, loaders["val"], device)
        test = evaluate(cfg, model, loaders["test"], device)

        theta_scheduler.step()
        # Only advance phi's schedule when phi actually stepped. Stepping a
        # scheduler whose optimizer never ran is meaningless and torch warns
        # about it on every epoch.
        if phi_scheduler is not None and cfg.needs_meta_step:
            phi_scheduler.step()

        record = {
            "epoch": epoch,
            "inner_lr": inner_lr,
            "train_loss_primary": train_stats["loss_primary"],
            "train_loss_auxiliary": train_stats["loss_auxiliary"],
            "train_acc_primary": train_stats["acc_primary"],
            "train_acc_auxiliary": train_stats["acc_auxiliary"],
            "train_cosine": train_stats["cosine"],
            "train_weight_mean": train_stats["weight_mean"],
            "train_weight_std": train_stats["weight_std"],
            "train_weight_min": train_stats["weight_min"],
            "train_weight_max": train_stats["weight_max"],
            "meta_loss_before": meta_stats.get("loss_before"),
            "meta_loss_after": meta_stats.get("loss_after"),
            "meta_entropy": meta_stats.get("entropy"),
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "test_loss": test["loss"],
            "test_acc": test["acc"],
            "test_aux_vs_fine_acc": test["aux_vs_fine_acc"],
            "elapsed_s": round(time.time() - started, 1),
        }
        logger.metric(record)
        # Mirrored, not forwarded: metrics.jsonl is written first and is what
        # the analysis reads, so this line failing costs visibility, not data.
        tracker.log(record, step=epoch)
        logger.print(
            f"epoch {epoch:03d} | train {train_stats['loss_primary']:.4f}/"
            f"{train_stats['acc_primary']:.4f} | val {val['acc']:.4f} "
            f"| test {test['acc']:.4f}"
        )

        if epoch in dump_epochs and train_stats["weights"] is not None:
            np.savez_compressed(
                f"{save_path}/dump_epoch{epoch:03d}.npz",
                weights=train_stats["weights"],
                sublabels=train_stats["sublabels"],
                fine=train_stats["fine_labels"],
                coarse=train_stats["coarse_labels"],
            )

        # Selection on validation. `train_wamal_network` selects on test, which
        # is test-set selection; CLINC150 ships a validation split so there is
        # no reason to inherit that here.
        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            best_epoch = epoch
            best_test_acc = test["acc"]
            if cfg.save_checkpoint:
                model.save(f"{save_path}/best_model.pt", epoch=epoch,
                           val_acc=val["acc"], test_acc=test["acc"])

    # After the loop, so it describes the phi that finished training. Only the arms
    # that generate labels have anything to record.
    if cfg.uses_generated_labels:
        sub, primary = test_assignment(cfg, model, loaders["test"], device)
        np.savez_compressed(f"{save_path}/test_assignment.npz",
                            sublabels=sub, primary=primary)

    summary = {
        "cell": cfg.cell_id(),
        "arm": cfg.arm,
        "dataset": cfg.dataset,
        "model": cfg.model,
        "shots": cfg.shots,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc_at_best_val": best_test_acc,
        "final_test_acc": test["acc"],
        "final_test_aux_vs_fine_acc": test["aux_vs_fine_acc"],
        "epochs": cfg.total_epoch,
        "train_size": train_size,
        "wall_time_s": round(time.time() - started, 1),
        "peak_memory_mb": (
            round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
            if device.type == "cuda" else None
        ),
        "trainable": model.memory_report(),
    }
    logger.print(f"done: best val {best_val_acc:.4f} @ epoch {best_epoch}, "
                 f"test {best_test_acc:.4f}, {summary['wall_time_s']}s")
    tracker.summarise(summary)
    mark_complete(save_path, summary)
    return summary


def _dump_schedule(cfg) -> set[int]:
    if not cfg.needs_label_network:
        return set()
    epochs = {0, cfg.total_epoch - 1}
    if cfg.dump_every > 0:
        epochs |= set(range(0, cfg.total_epoch, cfg.dump_every))
    return epochs


def _train_epoch(cfg, model, loader, device, optimizer, inner_lr,
                 collect_dumps: bool, train_size: int) -> dict:
    """Phase A: update theta on the (optionally weighted) multi-task loss."""
    model.train()

    loss_p_sum = loss_a_sum = 0.0
    correct_p = correct_a = total = 0
    cosine_sum, cosine_n = 0.0, 0
    # Sum of squares and extremes as well as the sum: the mean weight says nothing
    # about whether the weights differ *between samples*, which is the entire
    # difference between WAMAL and MAXL. A constant weight is MAXL with a rescaled
    # auxiliary loss, and the mean alone cannot distinguish the two.
    weight_sum = weight_sq_sum = 0.0
    weight_n = 0
    weight_min, weight_max = float("inf"), float("-inf")

    weights_out = np.full(train_size, np.nan, dtype=np.float32) if collect_dumps else None
    sublabels_out = np.full(train_size, -1, dtype=np.int64) if collect_dumps else None
    fine_out = np.full(train_size, -1, dtype=np.int64) if collect_dumps else None
    coarse_out = np.full(train_size, -1, dtype=np.int64) if collect_dumps else None

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        n = batch["coarse"].size(0)

        primary, auxiliary = model(batch["input_ids"], batch["attention_mask"],
                                   mode="primary")
        primary_loss = model_fit(primary, batch["coarse"], device, pri=True,
                                 num_output=cfg.num_primary)

        aux_target, raw_weights = (None, None)
        auxiliary_loss = torch.zeros_like(primary_loss)
        if cfg.uses_auxiliary:
            aux_target, raw_weights = _auxiliary_targets(cfg, model, batch, device,
                                                         detach=True)
            auxiliary_loss = model_fit(auxiliary, aux_target, device, pri=False,
                                       num_output=cfg.num_auxiliary)

        optimizer.zero_grad(set_to_none=True)

        if cfg.log_cosine and cfg.uses_auxiliary:
            cosine_sum += _cosine_similarity(model, primary_loss, auxiliary_loss)
            cosine_n += 1

        total_loss, aux_term = _combine(cfg, primary_loss, auxiliary_loss, raw_weights)
        total_loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.theta_parameters(), cfg.grad_clip)
        optimizer.step()

        loss_p_sum += primary_loss.sum().item()
        loss_a_sum += float(aux_term.item()) * n
        correct_p += primary.argmax(dim=1).eq(batch["coarse"]).sum().item()
        if aux_target is not None:
            correct_a += auxiliary.argmax(dim=1).eq(aux_target.argmax(dim=1)).sum().item()
        total += n

        if raw_weights is not None and cfg.uses_learned_weights:
            flat = raw_weights.detach().float().reshape(-1)
            weight_sum += flat.sum().item()
            weight_sq_sum += flat.square().sum().item()
            weight_min = min(weight_min, flat.min().item())
            weight_max = max(weight_max, flat.max().item())
            weight_n += n

        if collect_dumps and aux_target is not None:
            index = batch["index"].cpu().numpy()
            # Only for arms that actually use the weights. Under maxl the weight
            # head is never trained -- nothing in the loss touches it, so it
            # keeps its initialisation -- and dumping its output would put
            # random numbers in a file the weight analysis reads.
            if raw_weights is not None and cfg.uses_learned_weights:
                weights_out[index] = raw_weights.squeeze(-1).float().cpu().numpy()
            sublabels_out[index] = aux_target.argmax(dim=1).cpu().numpy()
            fine_out[index] = batch["fine"].cpu().numpy()
            coarse_out[index] = batch["coarse"].cpu().numpy()

    denom = max(total, 1)
    weight_mean = weight_sum / weight_n if weight_n else None
    weight_std = None
    if weight_n:
        # Clamped at zero because sum-of-squares minus square-of-sum is only
        # non-negative up to float error, and a tiny negative here is exactly the
        # case that matters -- genuinely constant weights.
        variance = max(weight_sq_sum / weight_n - weight_mean ** 2, 0.0)
        weight_std = variance ** 0.5
    return {
        "loss_primary": loss_p_sum / denom,
        "loss_auxiliary": loss_a_sum / denom,
        "acc_primary": correct_p / denom,
        "acc_auxiliary": correct_a / denom,
        "cosine": cosine_sum / cosine_n if cosine_n else None,
        "weight_mean": weight_mean,
        "weight_std": weight_std,
        "weight_min": weight_min if weight_n else None,
        "weight_max": weight_max if weight_n else None,
        "weights": weights_out,
        "sublabels": sublabels_out,
        "fine_labels": fine_out,
        "coarse_labels": coarse_out,
    }


def _query_batches(loader):
    """Endless stream of query batches, restarting the loader when exhausted.

    The query split is much smaller than the support split, so one training epoch
    consumes it several times over. Built with `num_workers=0` upstream precisely
    so that restarting is cheap.
    """
    while True:
        empty = True
        for batch in loader:
            empty = False
            yield batch
        if empty:
            raise RuntimeError("meta query loader is empty")


def _meta_epoch(cfg, model, loader, device, phi_optimizer, inner_lr,
                query_loader=None) -> dict:
    """Phase B: update phi against the primary loss measured at `theta+`.

    With no `query_loader` the outer loss is measured on the same batch that drove
    the inner update, which is paper-faithful and what phase 1 ran. Its weakness
    is that once theta has memorised the training set that loss is ~0 and phi
    receives no gradient -- measured at 1e-5 by the halfway point of a k=5 run.

    With a `query_loader`, the inner update still comes from the support batch
    (phi can only act through the auxiliary loss on data theta trains on) but the
    outer loss is measured on withheld data. That cannot be driven to zero by
    memorisation, so phi keeps receiving signal for the whole run.
    """
    model.train()

    before_sum = after_sum = entropy_sum = 0.0
    total = outer_total = 0
    queries = _query_batches(query_loader) if query_loader is not None else None

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        n = batch["coarse"].size(0)

        primary, auxiliary = model(batch["input_ids"], batch["attention_mask"],
                                   mode="primary")
        # Not detached here: the whole point is that theta+ depends on phi.
        aux_target, raw_weights = _auxiliary_targets(cfg, model, batch, device,
                                                     detach=False)

        primary_loss = model_fit(primary, batch["coarse"], device, pri=True,
                                 num_output=cfg.num_primary)
        auxiliary_loss = model_fit(auxiliary, aux_target, device, pri=False,
                                   num_output=cfg.num_auxiliary)
        entropy = model_entropy(aux_target)

        phi_optimizer.zero_grad(set_to_none=True)
        total_loss, _ = _combine(cfg, primary_loss, auxiliary_loss, raw_weights)

        fast_weights = inner_sgd_update(model, total_loss, inner_lr,
                                        first_order=cfg.first_order)

        # Where the outer loss is measured. Under holdout this is withheld data,
        # so `before`/`after` describe generalisation rather than fit, and the
        # pair stays informative after theta has memorised the support split.
        outer = batch
        if queries is not None:
            outer = {k: v.to(device, non_blocking=True)
                     for k, v in next(queries).items()}
            with torch.no_grad():
                pre_primary, _ = model.forward_primary(
                    outer["input_ids"], outer["attention_mask"]
                )
                primary_loss = model_fit(pre_primary, outer["coarse"], device,
                                         pri=True, num_output=cfg.num_primary)

        post_primary, _ = model.forward_primary(
            outer["input_ids"], outer["attention_mask"], params=fast_weights
        )
        post_loss = model_fit(post_primary, outer["coarse"], device, pri=True,
                              num_output=cfg.num_primary)

        objective = post_loss.mean() + cfg.entropy_loss_factor * entropy
        objective.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.phi_parameters(), cfg.grad_clip)
        phi_optimizer.step()

        # theta accumulated gradients from the outer backward as a side effect;
        # drop them so they cannot leak into phase A of the next epoch.
        for param in model.theta_parameters():
            param.grad = None

        # Two counters, because under holdout the outer loss is measured on a
        # query batch whose size need not match the support batch. Averaging both
        # over the same denominator would silently misreport the loss pair by the
        # ratio of the two split sizes.
        before_sum += primary_loss.sum().item()
        after_sum += post_loss.sum().item()
        outer_total += outer["coarse"].size(0)
        entropy_sum += entropy.item() * n
        total += n

    denom = max(total, 1)
    outer_denom = max(outer_total, 1)
    return {
        "loss_before": before_sum / outer_denom,
        "loss_after": after_sum / outer_denom,
        "entropy": entropy_sum / denom,
    }
