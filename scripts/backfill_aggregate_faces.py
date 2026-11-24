"""为已完成的聚合结果补跑人脸识别（复用已保存的目标检测结果）。

背景：批量聚合一开始把 face 写死成 False，历史记录里只有 head 框没有姓名。
本脚本不改动目标检测结果，只对已保存的 <batch>/sources/<原图> 跑一次
「SCRFD 检脸 + ArcFace 1:N + 姓名贴到最贴合人头」，然后把
faces / steps.face / summary 写回 record.json，并重画结果图。

用法（在项目根目录执行）：
  .venv\\Scripts\\python.exe scripts\\backfill_aggregate_faces.py --folder 脸加姿态
  .venv\\Scripts\\python.exe scripts\\backfill_aggregate_faces.py --record <batch32>-<index>
  .venv\\Scripts\\python.exe scripts\\backfill_aggregate_faces.py --folder 脸加姿态 --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.aggregation import aggregate, aggregate_batch, aggregate_history  # noqa: E402


def folder_records(names: list[str]) -> list[str]:
    """把文件夹名（含子文件夹）展开成 record_id 列表。"""
    items = aggregate_history.folders()
    roots = [item['id'] for item in items if item['name'] in names]
    missing = sorted(set(names) - {item['name'] for item in items if item['id'] in roots})
    if missing:
        raise SystemExit(f"找不到文件夹：{'、'.join(missing)}")
    branch = aggregate_history.descendant_ids(items, roots)
    ids: list[str] = []
    for item in items:
        if item['id'] in branch:
            ids.extend(item['record_ids'])
    return list(dict.fromkeys(ids))


def summary_for(faces: list[dict], targets: list[dict]) -> dict:
    return {
        'named_targets': sum(1 for target in targets if target.get('name')),
        'faces': len(faces),
        'bound_faces': sum(1 for face in faces if face.get('target_index') is not None),
        'matched_faces': sum(1 for face in faces if face.get('status') == 'matched'),
        'unknown_faces': sum(1 for face in faces if face.get('status') == 'unknown'),
        'rejected_faces': sum(1 for face in faces if face.get('status') == 'rejected'),
    }


def apply_face(record_id: str, *, force: bool, dry_run: bool) -> dict:
    """对一条聚合记录补跑人脸识别，返回统计信息。"""
    batch_id, index_text = record_id.rsplit('-', 1)
    index = int(index_text)
    directory = aggregate_batch.directory(batch_id)
    manifest = directory / 'record.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'批量记录不存在：{manifest}')
    batch = json.loads(manifest.read_text(encoding='utf-8'))
    row = batch['results'][index]
    if row.get('status') != 'completed':
        raise ValueError(f'记录未完成（{row.get("status")}）：{record_id}')
    result = row['result']
    if (result.get('steps', {}).get('face') or {}).get('enabled') and not force:
        return {'record_id': record_id, 'skipped': '已有人脸识别结果'}

    snapshot = directory / 'sources' / row.get('source_file', row['source'])
    if not snapshot.is_file():
        raise FileNotFoundError(f'原图快照不存在：{snapshot}')

    image = aggregate.read_image(snapshot)
    targets = result.get('targets') or []
    outcome = aggregate.recognize_and_bind(image, targets)
    faces = outcome['faces']

    result['targets'] = targets
    result['faces'] = faces
    result['steps']['face'] = {**outcome['step'], 'source': 'backfill'}
    result['unbound_faces'] = [face['face_id'] for face in faces if face.get('target_index') is None]
    result.setdefault('summary', {}).update(summary_for(faces, targets))

    stats = {'record_id': record_id, 'source': row['source'], 'faces': len(faces),
             'named_targets': outcome['named'], 'step': outcome['step']}
    if dry_run:
        return {**stats, 'dry_run': True}

    # 重画结果图：目标框沿用已保存的检测结果，只是把姓名标签补上
    rendered = aggregate.render(image, targets)
    rendered.save(directory / 'results' / result['result'])
    delivered = aggregate.RESULT_DIR / result['result']
    if delivered.exists():
        shutil.copyfile(directory / 'results' / result['result'], delivered)

    temp = manifest.with_suffix('.json.tmp')
    temp.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(manifest)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description='为已完成的聚合结果补跑人脸识别')
    parser.add_argument('--folder', action='append', default=[], help='聚合结果文件夹名（含子文件夹）')
    parser.add_argument('--record', action='append', default=[], help='记录 ID，形如 <32位批次ID>-<序号>')
    parser.add_argument('--force', action='store_true', help='已有人脸识别结果的记录也重跑')
    parser.add_argument('--dry-run', action='store_true', help='只统计不写盘')
    args = parser.parse_args()

    record_ids = args.record + (folder_records(args.folder) if args.folder else [])
    record_ids = list(dict.fromkeys(record_ids))
    if not record_ids:
        parser.error('请用 --folder 或 --record 指定要处理的结果')

    print(f'待处理 {len(record_ids)} 条记录{"（dry-run）" if args.dry_run else ""}')
    started = time.perf_counter()
    done = failed = 0
    for record_id in record_ids:
        try:
            stats = apply_face(record_id, force=args.force, dry_run=args.dry_run)
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f'  [失败] {record_id}：{error}')
            continue
        if 'skipped' in stats:
            print(f'  [跳过] {record_id}：{stats["skipped"]}')
            continue
        done += 1
        step = stats.get('step') or {}
        print(f'  [完成] {stats.get("source") or record_id}：检出 {step.get("face_count", 0)} 张脸 · '
              f'识别 {step.get("matched", 0)} · 贴名 {stats.get("named_targets", 0)} 个人头 · '
              f'{step.get("elapsed_seconds", 0)}s')
    print(f'结束：成功 {done} 条，失败 {failed} 条，耗时 {round(time.perf_counter()-started, 1)}s')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
