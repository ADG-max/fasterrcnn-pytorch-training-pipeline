import numpy as np

LABEL_TEXT_COLOR = (255, 255, 255)

def get_visual_config(classes):
    """
    Auto-generate COLORS, LABEL_BG, LABEL_TEXT_COLOR
    based on CLASSES.
    """

    colors = []
    label_bg = {}

    for idx, cls in enumerate(classes):
        if cls == "__background__":
            colors.append([0.0, 0.0, 0.0])

        elif cls == "fire":
            colors.append([1.0, 0.0, 0.0])
            label_bg[cls] = (120, 0, 0)

        elif cls == "smoke":
            colors.append([0.0, 1.0, 1.0])
            label_bg[cls] = (0, 120, 120)

        elif cls == "other":
            colors.append([1.0, 1.0, 0.0])
            label_bg[cls] = (120, 120, 0)

        elif cls == "fire_smoke":
            colors.append([1.0, 0.3, 0.0])
            label_bg[cls] = (140, 70, 0)

        else:
            # fallback otomatis (tidak crash)
            np.random.seed(abs(hash(cls)) % 10**6)
            rand_color = np.random.rand(3).tolist()
            colors.append(rand_color)
            label_bg[cls] = tuple((np.array(rand_color) * 120).astype(int))

    return (
        np.array(colors),
        label_bg,
        LABEL_TEXT_COLOR
    )
