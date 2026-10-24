def clean_image_transforms(config):
    training = transforms.Compose([
        transforms.RandomResizedCrop(
            448, scale=(0.8, 1.0), ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.1),
        transforms.RandomRotation(10, interpolation=InterpolationMode.BICUBIC),
        transforms.ColorJitter(0.15, 0.15, 0.10, 0.02),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
    ])
    validation = timm.data.create_transform(**config, is_training=False)
    padded = transforms.Compose([
        CleanWholeImagePad(448, config["mean"]),
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
    ])
    return training, validation, padded


def clean_worker_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def clean_optimizer(model, output_dir):
    settings = mroot_clean_settings
    head_parameters = {id(parameter) for parameter in model.get_classifier().parameters()}
    depth = len(model.blocks)
    groups = {}
    records = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if id(parameter) in head_parameters:
            layer = "head"
            learning_rate = settings["head_lr"]
        else:
            match = re.match(r"blocks\.(\d+)\.", name)
            if match:
                layer_id = int(match.group(1)) + 1
            elif name.startswith(("norm.", "fc_norm.")):
                layer_id = depth
            else:
                layer_id = 0
            layer = str(layer_id)
            learning_rate = settings["backbone_lr"] * settings["layer_decay"] ** (depth - layer_id)

        key = (layer, learning_rate)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": learning_rate,
                "weight_decay": settings["weight_decay"],
            }
        groups[key]["params"].append(parameter)
        records.append({
            "name": name,
            "layer": layer,
            "max_lr": learning_rate,
            "weight_decay": settings["weight_decay"],
            "parameters": parameter.numel(),
        })

    parameters = [parameter for group in groups.values() for parameter in group["params"]]
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert sum(parameter.numel() for parameter in parameters) == sum(
        parameter.numel() for parameter in model.parameters()
    )

    pd.DataFrame(records).to_csv(output_dir / "parameter_lr_groups.csv", index=False)
    optimizer = torch.optim.AdamW(list(groups.values()), weight_decay=settings["weight_decay"])
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=3)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["max_epochs"] - 3, eta_min=settings["minimum_lr"]
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[3]
    )
    return optimizer, scheduler


def predict_clean_validation(runner, loader, device, fixed_tta=False):
    runner.eval()
    predictions = []
    labels = []

    with torch.inference_mode():
        for batch in loader:
            if fixed_tta:
                normal, padded, target = batch
                padded = padded.to(device, non_blocking=True)
            else:
                normal, target = batch

            normal = normal.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                normal_logits = runner(normal)
                if fixed_tta:
                    flip_logits = runner(torch.flip(normal, dims=[3]))
                    pad_logits = runner(padded)

            probability = normal_logits.float().softmax(1)
            if fixed_tta:
                probability = (
                    0.75 * (probability + flip_logits.float().softmax(1)) / 2
                    + 0.25 * pad_logits.float().softmax(1)
                )
            predictions.append(probability.cpu().numpy())
            labels.append(target.numpy())

    return np.concatenate(predictions), np.concatenate(labels)


def clean_prediction_metrics(labels, probabilities):
    assert probabilities.shape == (len(labels), 39)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(1), 1, atol=1e-5)
    ranks = (np.argsort(-probabilities, axis=1) == labels[:, None]).argmax(1) + 1
    return {
        "macro_map": float(average_precision_score(np.eye(39)[labels], probabilities, average="macro")),
        "top1": float((probabilities.argmax(1) == labels).mean()),
        "top5": float((ranks <= 5).mean()),
        "images": len(labels),
    }


def run_clean_training_arm(arm_name, use_mixed_examples):
    import shutil

    settings = dict(mroot_clean_settings)
    splits = mroot_clean_splits
    output_dir = splits["directory"] / arm_name
    output_dir.mkdir(exist_ok=False)

    train = splits["train"].reset_index(drop=True).copy()
    validation = splits["validation"].reset_index(drop=True).copy()
    classes = list(mroot_restored["classes"])
    mapping = dict(mroot_restored["mapping"])
    train_manifest = Path(splits["summary"]["train_manifest"])
    val_manifest = Path(splits["summary"]["validation_manifest"])

    assert hashlib.sha256(train_manifest.read_bytes()).hexdigest() == splits["summary"]["train_sha256"]
    assert hashlib.sha256(val_manifest.read_bytes()).hexdigest() == splits["summary"]["validation_sha256"]
    assert len(train) == 6790 and len(validation) == 388
    assert len(train) % 2 == 0
    assert set(train.label) == set(validation.label) == set(range(39))
    assert not set(train.duplicate_group) & set(validation.duplicate_group)
    assert shutil.disk_usage(output_dir).free > 2 * 1024**3

    seed_clean_run(settings["seed"])
    device = torch.device("cuda:0")
    gpu_ids = list(range(min(2, torch.cuda.device_count())))
    assert gpu_ids, "The existing Kaggle GPU session is required."
    micro_batch = 8 if len(gpu_ids) == 2 else 4
    accumulation = settings["effective_batch_size"] // micro_batch

    initial_path = Path(mroot_restored["base_initial_state"])
    initial_sha = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    assert initial_sha == "2ad075bccc2241b2b6a253729a3c63571d9005102e1a80a06bfd1efe0266f810"
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    assert initial["class_to_idx"] == mapping
    assert list(initial["classes"]) == classes

    architecture = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
    model = timm.create_model(architecture, pretrained=False, num_classes=39)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in initial["model_state_dict"].items()
    )
    data_config = dict(initial["data_cfg"])
    del initial
    model = model.to(device)

    train_transform, val_transform, pad_transform = clean_image_transforms(data_config)
    assert str(train_transform) == mroot_restored["base_result"]["augmentation"]
    counts = train.groupby("label").size().reindex(range(39)).to_numpy()
    sample_weights = torch.as_tensor(1.0 / counts[train.label.to_numpy()], dtype=torch.double)
    sampler_generator = torch.Generator().manual_seed(settings["seed"])
    loader_generator = torch.Generator().manual_seed(settings["seed"] + 1)
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train), replacement=True,
        generator=sampler_generator,
    )

    train_loader = DataLoader(
        CleanLeafImages(train, train_transform), batch_size=micro_batch,
        sampler=sampler, num_workers=2, pin_memory=True,
        persistent_workers=True, worker_init_fn=clean_worker_seed,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        CleanLeafImages(validation, val_transform), batch_size=micro_batch,
        shuffle=False, num_workers=2, pin_memory=True,
        persistent_workers=True,
    )
    pair_loader = DataLoader(
        CleanLeafImages(validation, val_transform, pad_transform),
        batch_size=micro_batch, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=False,
    )

    optimizer, scheduler = clean_optimizer(model, output_dir)
    previous_groups = pd.read_csv(
        Path(mroot_lrthird_result["checkpoint_path"]).parent / "parameter_lr_groups.csv"
    )
    current_groups = pd.read_csv(output_dir / "parameter_lr_groups.csv")
    group_comparison = current_groups.merge(
        previous_groups, on="name", suffixes=("_new", "_previous"),
        validate="one_to_one",
    )
    assert len(group_comparison) == len(current_groups) == len(previous_groups)
    assert np.allclose(group_comparison.max_lr_new, group_comparison.max_lr_previous)
    assert np.array_equal(group_comparison.parameters_new, group_comparison.parameters_previous)

    mixer = None
    if use_mixed_examples:
        mixer = Mixup(
            mixup_alpha=settings["mixup_alpha"],
            cutmix_alpha=settings["cutmix_alpha"],
            prob=settings["mix_probability"],
            switch_prob=settings["cutmix_switch_probability"],
            mode="batch", label_smoothing=settings["label_smoothing"],
            num_classes=39,
        )

    runner = torch.nn.DataParallel(model, device_ids=gpu_ids) if len(gpu_ids) > 1 else model
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    config = {
        **settings,
        "arm": arm_name,
        "mixed_examples": use_mixed_examples,
        "architecture": architecture,
        "initial_state": str(initial_path),
        "initial_sha256": initial_sha,
        "train_manifest": str(train_manifest),
        "validation_manifest": str(val_manifest),
        "train_manifest_sha256": splits["summary"]["train_sha256"],
        "validation_manifest_sha256": splits["summary"]["validation_sha256"],
        "training_images": len(train),
        "validation_images": len(validation),
        "sampler": "Class-balanced replacement sampling, 6790 draws per epoch",
        "augmentation": str(train_transform),
        "scheduler": "3-epoch warmup followed by 17-epoch cosine decay",
        "loss": "Mixed soft-target CE" if mixer else "CE with label smoothing 0.05",
        "micro_batch": micro_batch,
        "gradient_accumulation": accumulation,
        "gpu_ids": gpu_ids,
        "data_cfg": data_config,
        "class_to_idx": mapping,
        "classes": classes,
        "sources": [
            "https://arxiv.org/abs/1710.09412",
            "https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html",
        ],
    }
    with (output_dir / "experiment_config.json").open("x") as handle:
        json.dump(config, handle, indent=2)

    def write_status(**values):
        with (output_dir / "status.json").open("w") as handle:
            json.dump({"arm": arm_name, **values}, handle, indent=2)

    print(f"Starting {arm_name}: {len(train)} training, {len(validation)} validation images.", flush=True)
    print(f"Initial weights and every learning-rate group verified. Mixed examples: {use_mixed_examples}", flush=True)
    print("No final-test data is loaded. Best epoch is selected by normal validation mAP.", flush=True)

    history = []
    best_map = -1.0
    best_epoch = None
    best_path = None
    stale_epochs = 0
    overflow_count = 0
    started = time.perf_counter()
    stop_reason = "epoch_limit"
    seed_clean_run(settings["seed"])

    try:
        for epoch in range(1, settings["max_epochs"] + 1):
            elapsed = time.perf_counter() - started
            last_epoch_seconds = history[-1]["epoch_seconds"] if history else 0
            if history and elapsed + last_epoch_seconds > settings["hours_per_arm"] * 3600:
                stop_reason = "time_limit"
                break

            epoch_started = time.perf_counter()
            runner.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            seen = 0
            successful_updates = 0
            epoch_learning_rates = [group["lr"] for group in optimizer.param_groups]

            for step, (images, labels) in enumerate(train_loader, 1):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                batch_size = len(labels)
                group_start = ((step - 1) // accumulation) * accumulation * micro_batch
                group_size = min(settings["effective_batch_size"], len(train) - group_start)

                soft_targets = None
                if mixer is not None:
                    images, soft_targets = mixer(images, labels)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = runner(images)

                if soft_targets is None:
                    loss = F.cross_entropy(
                        logits.float(), labels,
                        label_smoothing=settings["label_smoothing"],
                    )
                else:
                    loss = -(soft_targets * F.log_softmax(logits.float(), dim=1)).sum(1).mean()

                assert torch.isfinite(loss), "Non-finite training loss."
                scaler.scale(loss * batch_size / group_size).backward()
                loss_total += float(loss.item()) * batch_size
                seen += batch_size

                if step % accumulation == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if torch.isfinite(gradient_norm):
                        successful_updates += 1
                    else:
                        overflow_count += 1
                        print(f"AMP backoff: {arm_name}, epoch {epoch}, count {overflow_count}", flush=True)
                        assert overflow_count <= 50, "Too many AMP overflows."
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                if step == 1 or step % 100 == 0 or step == len(train_loader):
                    print(
                        f"{arm_name} | epoch {epoch:02d} | batch {step}/{len(train_loader)} | "
                        f"loss {loss_total / seen:.5f} | {time.perf_counter() - epoch_started:.0f}s",
                        flush=True,
                    )
                    write_status(
                        phase="training", epoch=epoch, step=step,
                        total_steps=len(train_loader), training_rows_seen=seen,
                    )

            assert seen == len(train)
            probabilities, val_labels = predict_clean_validation(runner, val_loader, device)
            assert np.array_equal(val_labels, validation.label.to_numpy())
            score = clean_prediction_metrics(val_labels, probabilities)
            improved = score["macro_map"] > best_map + 1e-8

            row = {
                "epoch": epoch,
                "train_loss": loss_total / seen,
                "val_macro_map": score["macro_map"],
                "val_top1": score["top1"],
                "val_top5": score["top5"],
                "successful_updates": successful_updates,
                "amp_overflows_total": overflow_count,
                "max_lr": max(epoch_learning_rates),
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

            if improved:
                best_map = score["macro_map"]
                best_epoch = epoch
                stale_epochs = 0
                best_path = output_dir / f"{arm_name}_best_epoch{epoch:02d}_map{best_map:.6f}.pth"
                weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                with best_path.open("xb") as handle:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": weights,
                        "model_name": architecture,
                        "classes": classes,
                        "class_to_idx": mapping,
                        "data_cfg": data_config,
                        "seed": settings["seed"],
                        "val_map": best_map,
                        "val_acc": score["top1"],
                        "config": config,
                    }, handle)
                del weights
                np.savez_compressed(
                    output_dir / f"best_epoch{epoch:02d}_validation.npz",
                    probabilities=probabilities, labels=val_labels,
                    original_rows=validation.original_row.to_numpy(),
                )
            else:
                stale_epochs += 1

            print(
                f"{arm_name} | epoch {epoch:02d} validation | "
                f"mAP {score['macro_map']:.6f} | Top1 {score['top1']:.4f} | "
                f"Top5 {score['top5']:.4f} | best epoch {best_epoch} ({best_map:.6f})",
                flush=True,
            )
            scheduler.step()
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_checkpoint": str(best_path),
                "best_epoch": best_epoch,
                "best_map": best_map,
                "history": history,
                "config": config,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "sampler_generator_state": sampler_generator.get_state(),
                "loader_generator_state": loader_generator.get_state(),
            }, output_dir / "latest_resume.pth")
            write_status(
                phase="epoch_complete", epoch=epoch, metrics=score,
                best_epoch=best_epoch, best_map=best_map,
                best_checkpoint=str(best_path), stale_epochs=stale_epochs,
            )

            if stale_epochs >= settings["patience"]:
                stop_reason = "early_stopping"
                break

        assert best_path is not None and best_path.is_file()
        selected = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(selected["model_state_dict"], strict=True)
        del selected
        best_normal, val_labels = predict_clean_validation(runner, val_loader, device)
        best_fixed, fixed_labels = predict_clean_validation(runner, pair_loader, device, fixed_tta=True)
        assert np.array_equal(val_labels, fixed_labels)
        normal_score = clean_prediction_metrics(val_labels, best_normal)
        fixed_score = clean_prediction_metrics(val_labels, best_fixed)
        assert abs(normal_score["macro_map"] - best_map) < 1e-7

        np.savez_compressed(
            output_dir / "selected_validation_predictions.npz",
            normal=best_normal, fixed=best_fixed, labels=val_labels,
            original_rows=validation.original_row.to_numpy(),
            historical_copy_free=validation.historical_copy_free.to_numpy(),
            classes=np.array(classes),
        )
        result = {
            "status": "completed",
            "arm": arm_name,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "checkpoint_path": str(best_path),
            "checkpoint_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
            "normal": normal_score,
            "fixed_tta": fixed_score,
            "stop_reason": stop_reason,
            "amp_overflows": overflow_count,
            "elapsed_seconds": time.perf_counter() - started,
            "initial_sha256": initial_sha,
            "train_manifest_sha256": config["train_manifest_sha256"],
            "validation_manifest_sha256": config["validation_manifest_sha256"],
            "final_test_evaluated": False,
        }
        with (output_dir / "experiment_result.json").open("x") as handle:
            json.dump(result, handle, indent=2)
        write_status(phase="completed", result=result)
        print(f"Completed {arm_name}: {json.dumps(result)}", flush=True)

    except Exception as error:
        write_status(phase="failed", error=repr(error), history=history)
        raise

    finally:
        del runner, model, optimizer, scheduler, train_loader, val_loader, pair_loader
        gc.collect()
        torch.cuda.empty_cache()

    return result


def run_clean_paired_experiment():
    output_dir = mroot_clean_splits["directory"]
    original_hashes = {
        name: model_tensor_hash(model)
        for name, model in mroot_restored["models"].items()
    }
    original_manifest_hashes = {
        name: hashlib.sha256((mroot_restored["backup_root"] / name).read_bytes()).hexdigest()
        for name in ["CLEAN_train.csv", "CLEAN_val.csv"]
    }

    with (output_dir / "PAIR_STARTED.json").open("x") as handle:
        json.dump({
            "started": datetime.now(timezone.utc).isoformat(),
            "settings": mroot_clean_settings,
            "split_summary": mroot_clean_splits["summary"],
            "hypothesis": "MixUp/CutMix may reduce memorization beyond the gentler Base448 recipe.",
            "arms": ["control", "mixup_cutmix"],
            "bounded": "Only these two runs; at most 20 epochs and 2.5 hours per arm.",
            "test_set": "The 623-image final-test subset stays locked and is not read or evaluated.",
        }, handle, indent=2)

    results = {}
    for arm_name, mixed in [("control", False), ("mixup_cutmix", True)]:
        print(f"Authorized paired experiment: starting {arm_name}.", flush=True)
        results[arm_name] = run_clean_training_arm(arm_name, mixed)
        current_hashes = {
            name: model_tensor_hash(model)
            for name, model in mroot_restored["models"].items()
        }
        assert current_hashes == original_hashes
        print("Preservation check passed: all five existing models are unchanged.", flush=True)

    for name, expected in original_manifest_hashes.items():
        actual = hashlib.sha256((mroot_restored["backup_root"] / name).read_bytes()).hexdigest()
        assert actual == expected

    with (output_dir / "PAIRED_TRAINING_RESULTS.json").open("x") as handle:
        json.dump(results, handle, indent=2)
    print("Both authorized runs are complete. Ready for cached validation comparisons.", flush=True)
    return results
