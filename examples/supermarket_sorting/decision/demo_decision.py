from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ''}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from decision.task_manager import TaskManager
else:
    from .task_manager import TaskManager


def main() -> None:
    manager = TaskManager()
    manager.build_tasks_for_products(['kele', 'maidong', 'zhijin'])

    print('=== ranked plan ===')
    print(json.dumps(manager.export_plan(), indent=2, ensure_ascii=False))

    decision = manager.next_decision()
    print('=== first decision ===')
    print(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False))

    if decision.selected_task is not None:
        payload = manager.build_execution_payload(decision.selected_task)
        print('=== execution payload ===')
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
