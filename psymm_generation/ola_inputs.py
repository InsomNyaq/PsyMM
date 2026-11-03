"""Pure prompt utilities; no model loading or corpus mutation."""
import random

from .core import digest


def render_history(turn):
    text = f"Dialogue context: {turn['context']}\n"
    for utterance in turn.get('dialogue_history', []):
        name = {'speaker': 'Speaker', 'listener': 'Listener'}[utterance['role']]
        text += f"{name}: {utterance['utterance']}\n"
    return text


def fewshot_examples(pool, target_ids, count, seed):
    """Require a disjoint training pool, exclude every target dialogue and deduplicate."""
    if not count:
        return ''
    unique = {}
    for dialogue in pool:
        cid = str(dialogue['conversation_id'])
        if cid not in target_ids and dialogue.get('turns'):
            unique.setdefault(cid, dialogue)
    keys = sorted(unique)
    if len(keys) < count:
        raise ValueError('Insufficient disjoint training dialogues for few-shot examples')
    examples = []
    for cid in random.Random(seed).sample(keys, count):
        turn = unique[cid]['turns'][-1]
        examples.append(render_history(turn) + 'The response of Listener: ' + turn['response'])
    return 'Here are examples from a separate training pool:\n' + '\n\n'.join(examples) + '\n'
