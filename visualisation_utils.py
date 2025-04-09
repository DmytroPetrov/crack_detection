import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def plot_classification_results(paths_of_images, y_true, y_pred, number_of_images, result_type, y_proba=None):
    """
    Plots images that match a given classification result type (TP, FP, TN, FN).

    Args:
        paths_of_images (list): List of paths to images.
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        number_of_images (int): Number of images to plot.
        result_type (str): One of {'true_positive', 'false_positive', 'true_negative', 'false_negative'}.
        :param y_proba: list of probabilities of the anomaly
    """
    assert result_type in {'true_positive', 'false_positive', 'true_negative', 'false_negative'}, \
        "Invalid result_type. Must be one of 'true_positive', 'false_positive', 'true_negative', 'false_negative'."

    # Identify indices for each type
    matched_indices = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if result_type == 'true_positive' and true == 1 and pred == 1:
            matched_indices.append(i)
        elif result_type == 'false_positive' and true == 0 and pred == 1:
            matched_indices.append(i)
        elif result_type == 'true_negative' and true == 0 and pred == 0:
            matched_indices.append(i)
        elif result_type == 'false_negative' and true == 1 and pred == 0:
            matched_indices.append(i)

    # Limit to the number requested
    matched_indices = matched_indices[:number_of_images]
    n_images = len(matched_indices)

    if n_images == 0:
        print(f"No images found for result_type '{result_type}'")
        return

    n_cols = (n_images + 1) // 2
    fig, axs = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    axs = axs.flatten()

    for idx, img_idx in enumerate(matched_indices):
        img_path = paths_of_images[img_idx]
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        image = mpimg.imread(img_path)
        axs[idx].imshow(image)
        prob_in_title = f"Anom Conf: {round(y_proba[img_idx], 2)}" if y_proba else ""
        axs[idx].set_title(f"True: {y_true[img_idx]}, Pred: {y_pred[img_idx]}" +
                           prob_in_title)
        axs[idx].axis('off')

    # Hide unused subplots
    for j in range(len(matched_indices), len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()
