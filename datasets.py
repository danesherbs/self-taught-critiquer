# %%

import random
import collections
import typing
import torch

from typing import Dict, Sequence
from torch import Tensor

# %%

Question = collections.namedtuple("Question", ["x", "y"])
RationaleStep = collections.namedtuple("RationaleStep", ["x", "y", "acc", "carry", "step", "n_steps"])
Rationale = typing.List[RationaleStep]
Correction = collections.namedtuple("Correction", ["line", "actual_rs", "expected_rs"])
CorruptedExample = collections.namedtuple("CorruptedExample", [
    "question",
    "incorrect_rational",
    "incorrect_answer",
    "correction",
    "correction_rationale",
    "correction_answer",
])

# %%

def make_question(n_digits: int) -> Question:
    assert n_digits > 0, "need at least one digit to do arithmetic"
    min_digit = int("1" + "0" * (n_digits-1)) if n_digits > 1 else 0
    max_digit = int("9" * n_digits)
    x = random.randint(min_digit, max_digit)
    y = random.randint(min_digit, max_digit)
    return Question(x, y)

# %%

def make_rationale_step_corrupt(rs: RationaleStep) -> RationaleStep:
    corrupt_x, corrupt_y, corrupt_acc, corrupt_carry, corrupt_step, corrupt_n_steps = rs
    sample = random.uniform(0, 1)

    if 0 < sample <= 0.1:
        corrupt_x = corrupt_x + random.randint(1, 9)  # corrupt x
    elif 0.1 < sample <= 0.2:
        corrupt_y = corrupt_y + random.randint(1, 9)  # corrupt y
    elif 0.2 < sample <= 0.5:
        if corrupt_acc == "":
            corrupt_acc = "0"  # since acc might be empty string
        corrupt_acc = str(int(corrupt_acc) + random.randint(1, 9))  # corrupt acc
    else:
        corrupt_carry = 1 if corrupt_carry == 0 else 0  # corrupt carry

    return RationaleStep(
        x=corrupt_x,
        y=corrupt_y,
        acc=corrupt_acc,
        carry=corrupt_carry,
        step=corrupt_step,
        n_steps=corrupt_n_steps,
    )

# %%

def next_rational_step(rs: RationaleStep) -> RationaleStep:
    """
    7 2 + 5 ,  C: 0         step = 0, n_steps = 4
    7 + 0 , 7 C: 0          step = 1, n_steps = 4
    , 7 7 C: 1              step = 2, n_steps = 4
    1 7 7                   step = 3, n_steps = 4
    """
    if rs.step == rs.n_steps - 1:
        return rs

    unit_x = rs.x % 10
    unit_y = rs.y % 10
    unit_sum = unit_x + unit_y + rs.carry

    new_x = rs.x // 10
    new_y = rs.y // 10
    new_acc = f"{unit_sum % 10}{rs.acc}"
    new_carry = unit_sum // 10
    new_step = rs.step + 1
    new_n_steps = rs.n_steps

    return RationaleStep(
        x=new_x,
        y=new_y,
        acc=new_acc,
        carry=new_carry,
        step=new_step,
        n_steps=new_n_steps,
    )

# %%

def make_rationale(q: Question, is_corrupted=False) -> Rationale:
    n_steps = max(len(str(q.x)), len(str(q.y))) + 2
    corrupt_step = random.randint(0, n_steps-3) if is_corrupted else -1
    rationale = []
    rationale_step = RationaleStep(
        x=q.x,
        y=q.y,
        acc="",
        carry=0,
        step=0,
        n_steps=n_steps,
    )
    
    for i in range(n_steps):
        if corrupt_step == i:
            rationale_step = make_rationale_step_corrupt(rationale_step)   # accumulate mistakes

        rationale.append(rationale_step)
        rationale_step = next_rational_step(rationale_step)
    
    return rationale

# %%

def correct_rationale(q: Question, actual_r: Rationale) -> typing.Union[Correction, None]:
    """
    Loops through rationale steps, finding first that doesn't match expected.
    Returns None if no mistake is found.
    """
    if len(actual_r) == 0:
        return None
    
    expected_r = make_rationale(q, is_corrupted=False)

    for i, (actual_rs, expected_rs) in enumerate(zip(actual_r, expected_r)):
        if actual_rs == expected_rs:
            continue  # nothing to see here

        return Correction(
            line=i,
            actual_rs=actual_rs,
            expected_rs=expected_rs,
        )

    return None

# %%

def complete_rationale(rs: RationaleStep, is_corrupted=False) -> Rationale:
    n_remaining_steps = (rs.n_steps - 1) - rs.step

    assert n_remaining_steps > 0, "Rationale step is already complete."

    if is_corrupted:
        assert n_remaining_steps > 1, "Accumulator wont be changed if carry is corrupted in the final rationale step, so at least two remaining steps are needed to gaurantee corruption."

    rationale = [rs]
    corrupted_step = random.randint(0, n_remaining_steps-2) if is_corrupted else -1

    for i in range(n_remaining_steps):
        rs = next_rational_step(rs)

        if corrupted_step == i:
            rs = make_rationale_step_corrupt(rs)

        rationale.append(rs)
    
    return rationale

# %%

def make_corrupted_example(q: typing.Union[Question, None] = None, default_n_digits=3, is_correction_r_corrupted=False) -> CorruptedExample:
    if q is None:
        q = make_question(n_digits=default_n_digits)
    
    incorrect_r = make_rationale(q, is_corrupted=True)
    incorrect_a = int(incorrect_r[-1].acc)

    assert incorrect_a != q.x + q.y, f"Incorrect answer should not be correct! Got {incorrect_a} but expected {q.x + q.y}."
    assert correct_rationale(q, incorrect_r) is not None, f"Mistake must be present in corrupted example.\n{incorrect_r}."

    correction = correct_rationale(q, incorrect_r)

    assert correction is not None, f"Mistake must be present in corrupted example.\n{q}\n{incorrect_r}"

    correction_r = complete_rationale(correction.expected_rs, is_corrupted=is_correction_r_corrupted)
    correction_a = int(correction_r[-1].acc)
    
    if not is_correction_r_corrupted:
        assert correction_a == q.x + q.y, f"Correction answer should be correct! Got {correction_a} but expected {q.x + q.y}."
    
    if is_correction_r_corrupted:
        assert correction_a != q.x + q.y, f"Correction answer should not be correct! Got {correction_a} but expected something different from {q.x + q.y}."

    return CorruptedExample(
        question=q,
        incorrect_rational=incorrect_r,
        incorrect_answer=incorrect_a,
        correction=correction,
        correction_rationale=correction_r,
        correction_answer=correction_a,
    )

# %%

def number_to_str(x: int) -> str:
    return " ".join(list(str(x)))

# %%

def rationale_step_to_str(rs: RationaleStep) -> str:
    if rs.step == rs.n_steps - 1:
        return number_to_str(rs.acc)
    elif rs.step == rs.n_steps - 2:
        return f", {number_to_str(rs.acc)} C: {rs.carry}"
    else:
        return f"{number_to_str(rs.x)} + {number_to_str(rs.y)} , {number_to_str(rs.acc)} C: {rs.carry}"

# %%

def rationale_to_str(r: Rationale) -> str:
    return "\n".join(rationale_step_to_str(step) for step in r)

# %%

def correction_to_str_concise(c: Correction) -> str:
    if c.actual_rs.x != c.expected_rs.x:
        return f"line {c.line} : \"{c.actual_rs.x}\" should be \"{c.expected_rs.x}\""
    elif c.actual_rs.y != c.expected_rs.y:
        return f"line {c.line} : \"{c.actual_rs.y}\" should be \"{c.expected_rs.y}\""
    elif c.actual_rs.acc != c.expected_rs.acc:
        return f"line {c.line} : \"{c.actual_rs.acc}\" should be \"{c.expected_rs.acc}\""
    elif c.actual_rs.carry != c.expected_rs.carry:
        return f"line {c.line} : \"C: {c.actual_rs.carry}\" should be \"C: {c.expected_rs.carry}\""
    else:
        raise ValueError(f"Expected rational step is the same as actual! Got {c.actual_rs} for both actual and expected.")

# %%

def correction_to_str_verbose(c: Correction) -> str:
    if c.actual_rs == c.expected_rs:
        raise ValueError(f"Expected rational step is the same as actual! Got {c.actual_rs} for both actual and expected.")
    
    actual_rs_str = rationale_step_to_str(c.actual_rs)
    expected_rs_str = rationale_step_to_str(c.expected_rs)

    return f"\"{actual_rs_str}\" should be \"{expected_rs_str}\""

# %%

def corrupted_example_to_str(ce: CorruptedExample, include_rationale_in_critique=True) -> str:
    question, incorrect_r, incorrect_a, correction, correction_r, correction_a = ce
    x_str = number_to_str(question.x)
    y_str = number_to_str(question.y)
    incorrect_r_str = rationale_to_str(incorrect_r)
    incorrect_a_str = number_to_str(incorrect_a)
    correction_str = correction_to_str_verbose(correction)
    correction_r_str = rationale_to_str(correction_r)
    correction_a_str = number_to_str(correction_a)
    
    if not include_rationale_in_critique:
        return f"""Input:
{x_str} + {y_str}
Target:
<scratch>
{incorrect_r_str}
</scratch>
{incorrect_a_str}
Correction:
{correction_str}"""

    return f"""Input:
{x_str} + {y_str}
Target:
<scratch>
{incorrect_r_str}
</scratch>
{incorrect_a_str}
Correction:
{correction_str}
<scratch>
{correction_r_str}
</scratch>
{correction_a_str}"""

# %%

def corrupted_example_to_prompt_critique(ce: CorruptedExample, include_rationale_in_critique=True) -> typing.Tuple[str, str]:
    question, incorrect_r, incorrect_a, correction, correction_r, correction_a = ce
    x_str = number_to_str(question.x)
    y_str = number_to_str(question.y)
    incorrect_r_str = rationale_to_str(incorrect_r)
    incorrect_a_str = number_to_str(incorrect_a)
    correction_str = correction_to_str_verbose(correction)
    correction_r_str = rationale_to_str(correction_r)
    correction_a_str = number_to_str(correction_a)

    if not include_rationale_in_critique:
        correction_str = correction_to_str_verbose(correction)
        return (
f"""Input:
{x_str} + {y_str}
Target:
<scratch>
{incorrect_r_str}
</scratch>
{incorrect_a_str}
Correction:
""",

f"""{correction_str}""")
    
    return (
f"""Input:
{x_str} + {y_str}
Target:
<scratch>
{incorrect_r_str}
</scratch>
{incorrect_a_str}
Correction:
""",

f"""{correction_str}
<scratch>
{correction_r_str}
</scratch>
{correction_a_str}""")

# %%

def generate_few_shot_examples(min_n_digits=1, max_n_digits=5, n_examples=4, include_rationale_in_critique=False, random_seed=42):
    random.seed(random_seed)  # make dataset same each time
    few_shot_examples = ""
    
    for _ in range(n_examples):
        n_digits = random.randint(min_n_digits, max_n_digits)
        q = make_question(n_digits=n_digits)
        ce = make_corrupted_example(q)
        ce_str = corrupted_example_to_str(ce, include_rationale_in_critique=include_rationale_in_critique)
        few_shot_examples += ce_str + "\n\n"
    
    return few_shot_examples

# %%

def generate_labelled_few_shot_examples(min_n_digits=1, max_n_digits=5, n_examples=4, include_rationale_in_critique=False, random_seed=42):
    random.seed(random_seed)  # make dataset same each time
    few_shot_examples = ""

    for _ in range(n_examples):
        is_correction_corrupted = random.random() < 0.5
        label = "[FAIL]" if is_correction_corrupted else "[PASS]"
        n_digits = random.randint(min_n_digits, max_n_digits)
        q = make_question(n_digits=n_digits)
        ce = make_corrupted_example(q, is_correction_r_corrupted=is_correction_corrupted)
        ce_str = corrupted_example_to_str(ce, include_rationale_in_critique=include_rationale_in_critique)
        few_shot_examples += ce_str + f" {label}" + "\n\n"
    
    return few_shot_examples

# %%

class ArithmeticDataset(torch.utils.data.Dataset):
    def __init__(self, min_n_digits=1, max_n_digits=8, n_examples=10_000, random_seed=42, include_rationale_in_critique=True, mask=None):
        random.seed(random_seed)  # make dataset same each time
        self.n_examples = n_examples
        self.questions = [make_question(random.randint(min_n_digits, max_n_digits)) for _ in range(n_examples)]
        self.examples = [corrupted_example_to_prompt_critique(make_corrupted_example(q), include_rationale_in_critique=include_rationale_in_critique) for q in self.questions]

        if mask is not None:
            self.examples = [e for i, e in enumerate(self.examples) if mask[i]]

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# %%

class LabelledArithmeticDataset(torch.utils.data.Dataset):
    def __init__(self, min_n_digits=1, max_n_digits=1, n_examples=10_000, random_seed=42):
        random.seed(random_seed)  # make dataset same each time
        self.n_examples = n_examples
        self.questions = [make_question(random.randint(min_n_digits, max_n_digits)) for _ in range(n_examples)]
        self.corrupted_examples = [(make_corrupted_example(q, is_correction_r_corrupted=i%2==0), i%2!=0) for i, q in enumerate(self.questions)]
        self.prompt_and_completions = [(corrupted_example_to_prompt_critique(ce), label) for ce, label in self.corrupted_examples]
        self.examples = [(prompt + completion, label) for ((prompt, completion), label) in self.prompt_and_completions]

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# %%

class FewshotDiscriminatorArithmeticDataset(torch.utils.data.Dataset):
    def __init__(self, min_n_digits=1, max_n_digits=1, n_few_shot_examples=5, n_examples=10_000, random_seed=42):
        random.seed(random_seed)  # make dataset same each time
        self.n_examples = n_examples
        few_shot_examples = generate_labelled_few_shot_examples(
            min_n_digits=min_n_digits,
            max_n_digits=max_n_digits,
            n_examples=n_few_shot_examples,
            include_rationale_in_critique=True,
            random_seed=random_seed,
        )
        self.qs = [make_question(random.randint(min_n_digits, max_n_digits)) for _ in range(n_examples)]
        self.corrupted_examples = [(make_corrupted_example(q, is_correction_r_corrupted=i%2==0), i%2!=0) for i, q in enumerate(self.qs)]
        self.prompt_and_completions = [(corrupted_example_to_str(ce), label) for ce, label in self.corrupted_examples]
        self.examples = [(few_shot_examples + ce_str + " [", label) for (ce_str, label) in self.prompt_and_completions]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


# %%

class DictDataset(torch.utils.data.Dataset):
    """Makes a dataset from a dictionary of tensors."""

    def __init__(self, inputs: Dict[str, Tensor]):
        assert len(inputs) > 0, "inputs must be non-empty"
        
        keys = list(inputs.keys())
        key = keys[0]
        self._length = inputs[key].shape[0]
        
        for v in inputs.values():
            assert v.shape[0] == self._length, "all tensors must have same shape in first dimension"
        
        self._inputs = inputs

    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx) -> Dict[str, Tensor]:
        return {k: v[idx] for k, v in self._inputs.items()}

# %%

def dictdataset_collate_fn(batch: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Collate function for DictDataset."""

    assert len(batch) > 0, "batch must be non-empty"
    keys = list(batch[0].keys())
    return {k: torch.vstack([example[k] for example in batch]) for k in keys}

# %%
