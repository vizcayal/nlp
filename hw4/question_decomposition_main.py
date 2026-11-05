"""Local development driver for automatically scored Part 3."""

import argparse
import json
from pathlib import Path

from factcheck import QuestionDecompositionFactChecker


def load_tasks(path):
    tasks = []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        task = json.loads(line)
        if not isinstance(task, dict):
            raise ValueError(f'Question task on line {line_no} must be an object')
        choices = task.get('candidates')
        gold = task.get('gold_ids')
        if (not isinstance(task.get('claim'), str) or not task['claim'].strip()
                or not isinstance(choices, list) or len(choices) != 6):
            raise ValueError(f'Invalid question task on line {line_no}')
        if not all(isinstance(c, dict) and isinstance(c.get('id'), str)
                   and isinstance(c.get('question'), str) and c['question'].strip() for c in choices):
            raise ValueError(f'Invalid candidate objects on line {line_no}')
        ids = [candidate['id'] for candidate in choices]
        if (len(set(ids)) != 6 or not isinstance(gold, list) or len(gold) != 2
                or not all(isinstance(x, str) for x in gold) or len(set(gold)) != 2
                or not set(gold) <= set(ids)):
            raise ValueError(f'Invalid candidate IDs or key on line {line_no}')
        tasks.append(task)
    if not tasks:
        raise ValueError('Question task file is empty')
    return tasks


def evaluate(tasks, checker):
    correct = 0
    invalid = 0
    for task_no, task in enumerate(tasks, 1):
        candidates = [{'id': item['id'], 'question': item['question']} for item in task['candidates']]
        selected = checker.select_questions(task['claim'], candidates)
        valid = {item['id'] for item in candidates}
        if (not isinstance(selected, (list, tuple)) or len(selected) != 2
                or not all(isinstance(x, str) for x in selected)
                or len(set(selected)) != 2 or not set(selected) <= valid):
            invalid += 1
            print(f"Task {task_no}: invalid response, 0/2")
            continue
        hits = len(set(selected) & set(task['gold_ids']))
        correct += hits
        print(f"Task {task_no}: {hits}/2 required questions selected")
    score = correct / (2 * len(tasks))
    print(f'Question-selection F1: {score:.4f} ({correct}/{2 * len(tasks)} required questions)')
    print(f'Invalid responses: {invalid}')
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/questions_dev.jsonl')
    args = parser.parse_args()
    evaluate(load_tasks(args.data_path), QuestionDecompositionFactChecker())


if __name__ == '__main__':
    main()
