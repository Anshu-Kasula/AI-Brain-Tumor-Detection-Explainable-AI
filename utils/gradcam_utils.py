import tensorflow as tf
import numpy as np
import cv2

# ---------------- GRADCAM ---------------- #
def get_gradcam_heatmap(model, img_array):

    last_conv_layer_name = "conv5_block3_out"  # ResNet50

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, list):
            predictions = predictions[0]

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    print("Heatmap min:", np.min(heatmap))
    print("Heatmap max:", np.max(heatmap))
    print("Heatmap mean:", np.mean(heatmap))

    return heatmap


# ---------------- OVERLAY ---------------- #
def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(255 - heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(
        original_img,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return superimposed_img

