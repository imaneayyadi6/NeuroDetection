import cv2
import numpy as np

def segment_image(img):
    # Dummy segmentation: seuillage simple
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return mask

def apply_model_and_color(img, model):
    img_resized = cv2.resize(img, (150, 150))
    img_input = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_input, axis=(0, -1))  # (1,150,150,1)
    activation = model.predict(img_input)[0]

    heatmap = cv2.resize(activation[:, :, 0], (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, heatmap_color, 0.4, 0)

    return overlay, heatmap




# Ce code permet de traiter une image et de visualiser les zones importantes détectées par un modèle CNN. 
# La fonction `segment_image` applique un seuillage simple pour créer un masque binaire qui sépare les 
# pixels clairs et sombres, ce qui peut aider à isoler certaines régions de l’image. La fonction 
# `apply_model_and_color` redimensionne d’abord l’image à la taille attendue par le modèle (150x150), 
# puis normalise les valeurs des pixels avant de l’envoyer au modèle pour la prédiction. La sortie du 
# modèle est utilisée pour générer une heatmap représentant les activations importantes. Cette heatmap 
# est ensuite colorée et superposée à l’image originale afin de visualiser les zones de l’image qui ont 
# le plus influencé la décision du modèle.