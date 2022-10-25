from matplotlib import pyplot as plt


def draw_mask(prediction, title: str = "Figure", pyplot_block=True, show=True):
    """basic prediction visualization showing argmax masks and softmax masks if available"""
    masks = prediction.squeeze()
    fig, axes = plt.subplots(len(masks), num=title)
    for mask, ax in zip(masks, axes):
        ax.imshow(mask)
        ax.set_axis_off()

    if show:
        plt.show(block=pyplot_block)
