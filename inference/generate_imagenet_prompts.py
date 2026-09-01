"""Generate a large T2V prompt list aligned with the ImageNet-1k classes.

Inception Score uses an ImageNet-trained Inception v3, so prompts that each
depict one distinct ImageNet class maximize the diversity term of IS.

Output format matches prompts.txt: one double-quoted lowercase sentence per
line, ending with a period. Class names containing characters unsafe for the
derived output filename (comma, apostrophe, slash) are skipped, and the list
is truncated to a multiple of 12 so 1_parse_prompts.sh can split it evenly
into 2, 3, 4, 6 or 12 parts.
"""

import re

from torchvision.models import Inception_V3_Weights

OUTPUT_FILE = "prompts_imagenet.txt"
# Keep the count a multiple of 12 (LCM of 2/3/4/6) for 1_parse_prompts.sh
TARGET_MULTIPLE = 12

# Short, generation-friendly contexts cycled deterministically
TEMPLATES = [
    "{art} {name} in natural daylight.",
    "a close-up view of {art} {name}.",
    "{art} {name} outdoors on a sunny day.",
    "{art} {name} in a realistic setting.",
    "a detailed video of {art} {name}.",
    "{art} {name} shown clearly in the center of the frame.",
]

SAFE_NAME = re.compile(r"^[a-z0-9 -]+$")


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def main():
    categories = Inception_V3_Weights.IMAGENET1K_V1.meta["categories"]

    names = []
    seen = set()
    for cat in categories:
        name = cat.lower().strip()
        # Skip duplicates after lowercasing (e.g., "Cardigan" the sweater vs
        # the corgi) — duplicate prompts overwrite each other's output file
        if SAFE_NAME.match(name) and name not in seen:
            seen.add(name)
            names.append(name)

    usable = len(names) - (len(names) % TARGET_MULTIPLE)
    names = names[:usable]

    lines = []
    for i, name in enumerate(names):
        template = TEMPLATES[i % len(TEMPLATES)]
        lines.append('"%s"' % template.format(art=article(name), name=name))

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"classes total: {len(categories)}, usable: {usable}")
    print(f"wrote {len(lines)} prompts to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
