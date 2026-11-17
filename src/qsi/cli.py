"""로컬 실험 명령행 진입점."""

import argparse

import pandas as pd

from . import seed as seed_utils, config, experiment, io, report


def main(argv=None):
    """선택한 조건을 실행하고 결과 파일을 작성한다."""
    parser = argparse.ArgumentParser(description="Si 영역 분류 비교 실험")
    parser.add_argument("--conditions", default=",".join(c["key"] for c in config.CONDITIONS))
    parser.add_argument("--seeds", default=",".join(map(str, config.SEEDS)))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--backend", choices=("torch", "qiskit"), default="torch")
    parser.add_argument("--out", default=str(config.RESULTS_DIR))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    keys = [key.strip() for key in args.conditions.split(",")]
    known = {cond["key"]: cond for cond in config.CONDITIONS}
    unknown = set(keys) - set(known)
    if unknown:
        parser.error(f"알 수 없는 조건: {', '.join(sorted(unknown))}")
    seeds = tuple(int(value) for value in args.seeds.split(","))
    seed_utils.set_seed(seeds[0])  # 전역 난수도 고정(각 모델은 자기 시드로 다시 고정한다)
    train, test, overlap = io.split_by_alloy(io.load_data())
    thr_set = io.threshold_set(train)
    print(f"합금 분리 완료: 학습 {len(train)}행, 평가 {len(test)}행, 중복 합금 제외 {overlap}")
    rows, artifacts = [], {"_test": test}
    for key in keys:
        result = experiment.run_condition(known[key], train, test, thr_set, seeds, args.steps, args.backend)
        rows.extend(result)
        artifacts.update(result.models)
        for row in result:
            print(f"{row['key']} 시드 {row['seed']}: mIoU {row['miou']:.2f}, 학습 {row['train_seconds']:.2f}초")
    frame = pd.DataFrame(rows)
    verdicts = [experiment.verdict(frame, a, b) for a, b in config.VERDICT_PAIRS if a in keys and b in keys]
    report.write_results(frame, verdicts, artifacts, args.out, figures=not args.no_figures)
    print(f"요약 보고서: {args.out}/summary.md")


if __name__ == "__main__":
    main()
