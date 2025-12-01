"""
Pipeline YOLO + MobileNet pour la détection et classification des maladies de plantes
======================================================================================
- YOLO : Détecte et localise les feuilles dans l'image
- MobileNet : Classifie la maladie sur chaque feuille détectée
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path

# ========== CONFIGURATION ==========
# Chemins des modèles
YOLO_MODEL_PATH = "yolo11x_leaf.pt"  # Modèle YOLO entraîné pour détecter les feuilles
MOBILENET_MODEL_PATH = "plant_disease_model_optimized.h5"  # Meilleur modèle MobileNet (100% accuracy)
CLASS_LABELS_PATH = "class_labels.json"  # Fichier des labels de classes

# Paramètres
IMG_SIZE = (224, 224)  # Taille d'entrée pour MobileNet (plant_disease_model utilise 224x224)
YOLO_CONFIDENCE = 0.25  # Seuil de confiance YOLO
MOBILENET_CONFIDENCE = 0.5  # Seuil de confiance MobileNet
NUM_CLASSES = 38  # Nombre de classes de maladies


def build_mobilenet_model(num_classes):
    """Reconstruit le modèle MobileNet avec la même architecture que l'entraînement"""
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models

    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model


def load_models():
    """Charge les modèles YOLO et MobileNet"""
    from ultralytics import YOLO
    import tensorflow as tf

    print("📦 Chargement du modèle YOLO...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    print("📦 Chargement du modèle MobileNet...")
    # Charger le modèle complet (architecture + poids)
    mobilenet_model = tf.keras.models.load_model(MOBILENET_MODEL_PATH)

    # Charger les labels de classes
    print("📦 Chargement des labels de classes...")
    with open(CLASS_LABELS_PATH, 'r') as f:
        class_labels = json.load(f)

    print("✅ Tous les modèles sont chargés!\n")
    return yolo_model, mobilenet_model, class_labels


def preprocess_for_mobilenet(image):
    """Prétraite une image pour MobileNet"""
    # Redimensionner à 224x224
    resized = cv2.resize(image, IMG_SIZE)
    # Convertir BGR vers RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normaliser [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    # Ajouter la dimension batch
    batched = np.expand_dims(normalized, axis=0)
    return batched


def classify_image(image_path, mobilenet_model, class_labels, save_output=True):
    """
    Classification directe avec MobileNet (sans YOLO)
    Utilisez cette fonction quand les images contiennent déjà des feuilles cadrées.
    """
    # Charger l'image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Impossible de charger l'image: {image_path}")
        return [], None

    original_image = image.copy()

    print(f"🔍 Classification MobileNet sur {image_path}...")

    # Prétraiter et classifier
    preprocessed = preprocess_for_mobilenet(image)
    predictions = mobilenet_model.predict(preprocessed, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    class_name = class_labels.get(str(class_idx), "Unknown")

    result = {
        "class": class_name,
        "confidence": float(confidence)
    }

    # Annoter l'image
    color = (0, 255, 0) if "healthy" in class_name.lower() else (0, 0, 255)
    plant_disease = class_name.split('___')
    plant = plant_disease[0] if len(plant_disease) > 0 else "Unknown"
    disease = plant_disease[1] if len(plant_disease) > 1 else "Unknown"

    label1 = f"Plante: {plant}"
    label2 = f"Etat: {disease}"
    label3 = f"Confiance: {confidence:.2%}"

    cv2.putText(original_image, label1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(original_image, label2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(original_image, label3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Sauvegarder le résultat
    if save_output:
        output_path = Path(image_path).stem + "_result.jpg"
        cv2.imwrite(output_path, original_image)
        print(f"💾 Résultat sauvegardé: {output_path}")

    return [result], original_image


def detect_and_classify(image_path, yolo_model, mobilenet_model, class_labels, save_output=True):
    """
    Pipeline principal : Détection YOLO + Classification MobileNet

    Args:
        image_path: Chemin vers l'image
        yolo_model: Modèle YOLO chargé
        mobilenet_model: Modèle MobileNet chargé
        class_labels: Dictionnaire des labels
        save_output: Sauvegarder l'image annotée

    Returns:
        Liste des détections avec leurs classifications
    """
    # Charger l'image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Impossible de charger l'image: {image_path}")
        return [], None

    original_image = image.copy()
    results = []

    # Étape 1: Détection YOLO
    print(f"🔍 Détection YOLO sur {image_path}...")
    detections = yolo_model(image, conf=YOLO_CONFIDENCE, verbose=False)

    # Pour chaque détection (ou image entière si pas de détection spécifique)
    boxes = detections[0].boxes

    # Si pas de détection ou si YOLO détecte des objets non pertinents
    # On utilise directement MobileNet sur l'image entière
    if len(boxes) == 0:
        print("ℹ️ Aucune détection YOLO, classification de l'image entière...")
        return classify_image(image_path, mobilenet_model, class_labels, save_output)

    # Votre modèle YOLO est entraîné pour détecter les feuilles (classe 0 = "leaf")
    # Donc toutes les détections sont des feuilles !
    print(f"✅ {len(boxes)} feuille(s) détectée(s) par YOLO")

    # Pour chaque boîte détectée (feuilles)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        yolo_conf = float(box.conf[0])

        # Extraire la région d'intérêt (ROI)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Étape 2: Classification MobileNet sur le crop
        preprocessed = preprocess_for_mobilenet(crop)
        predictions = mobilenet_model.predict(preprocessed, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        class_name = class_labels.get(str(class_idx), "Unknown")

        results.append({
            "bbox": [x1, y1, x2, y2],
            "class": class_name,
            "confidence": float(confidence)
        })

        # Annoter l'image
        color = (0, 255, 0) if "healthy" in class_name.lower() else (0, 0, 255)
        cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)

        # Label avec classe et confiance
        label = f"{class_name.split('___')[-1]}: {confidence:.2%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(original_image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(original_image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Sauvegarder le résultat
    if save_output:
        output_path = Path(image_path).stem + "_result.jpg"
        cv2.imwrite(output_path, original_image)
        print(f"💾 Résultat sauvegardé: {output_path}")

    return results, original_image


def process_video(video_path, yolo_model, mobilenet_model, class_labels, output_path=None):
    """
    Traite une vidéo frame par frame
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la vidéo: {video_path}")
        return

    # Paramètres vidéo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Writer pour la vidéo de sortie
    if output_path is None:
        output_path = Path(video_path).stem + "_result.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    print(f"🎥 Traitement de la vidéo: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Sauvegarder temporairement la frame
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        # Traiter la frame
        _, annotated_frame = detect_and_classify(
            temp_path, yolo_model, mobilenet_model, class_labels, save_output=False
        )

        # Redimensionner si nécessaire
        if annotated_frame.shape[:2] != (height, width):
            annotated_frame = cv2.resize(annotated_frame, (width, height))

        out.write(annotated_frame)

        if frame_count % 30 == 0:
            print(f"  Frame {frame_count} traitée...")

    # Nettoyer
    cap.release()
    out.release()
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")

    print(f"✅ Vidéo sauvegardée: {output_path}")


def process_webcam(yolo_model, mobilenet_model, class_labels):
    """
    Traitement en temps réel via webcam
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam")
        return

    print("📷 Webcam active. Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sauvegarder temporairement la frame
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        # Traiter la frame
        results, annotated_frame = detect_and_classify(
            temp_path, yolo_model, mobilenet_model, class_labels, save_output=False
        )

        # Afficher les résultats
        cv2.imshow('Plant Disease Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")


def format_results(results):
    """Formate les résultats pour l'affichage"""
    print("\n" + "="*50)
    print("📊 RÉSULTATS DE L'ANALYSE")
    print("="*50)

    for i, result in enumerate(results):
        plant_disease = result['class'].split('___')
        plant = plant_disease[0] if len(plant_disease) > 0 else "Unknown"
        disease = plant_disease[1] if len(plant_disease) > 1 else "Unknown"

        print(f"\n🌿 Détection #{i+1}:")
        print(f"   Plante: {plant}")
        print(f"   État: {disease}")
        print(f"   Confiance: {result['confidence']:.2%}")

        if "healthy" in disease.lower():
            print("   ✅ Cette plante est en bonne santé!")
        else:
            print("   ⚠️ Maladie détectée - Traitement recommandé")

    print("\n" + "="*50)


# ========== FONCTION PRINCIPALE ==========
def main():
    """Fonction principale du pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Pipeline YOLO + MobileNet pour la détection de maladies de plantes')
    parser.add_argument('--image', type=str, help='Chemin vers une image à analyser')
    parser.add_argument('--video', type=str, help='Chemin vers une vidéo à analyser')
    parser.add_argument('--webcam', action='store_true', help='Utiliser la webcam')
    parser.add_argument('--folder', type=str, help='Dossier contenant des images à analyser')

    args = parser.parse_args()

    # Charger les modèles
    yolo_model, mobilenet_model, class_labels = load_models()

    if args.image:
        # Extraire le vrai label du chemin (nom du dossier parent)
        true_label = Path(args.image).parent.name

        # Traiter une image
        results, annotated_img = detect_and_classify(args.image, yolo_model, mobilenet_model, class_labels)
        format_results(results)

        # Afficher la comparaison avec le vrai label
        print(f"\n{'='*50}")
        print(f"📋 COMPARAISON AVEC LE VRAI LABEL")
        print(f"{'='*50}")
        print(f"🏷️  Vrai label    : {true_label}")
        if results:
            predicted = results[0]['class']
            conf = results[0]['confidence']
            is_correct = true_label.lower() == predicted.lower()
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            print(f"🤖 Prédiction    : {predicted} ({conf:.2%})")
            print(f"📊 Résultat      : {status}")
        print(f"{'='*50}")

        # Ouvrir l'image résultat
        result_path = Path(args.image).stem + "_result.jpg"
        if Path(result_path).exists():
            import subprocess
            subprocess.Popen(['start', '', result_path], shell=True)

    elif args.video:
        # Traiter une vidéo
        process_video(args.video, yolo_model, mobilenet_model, class_labels)

    elif args.webcam:
        # Traitement webcam
        process_webcam(yolo_model, mobilenet_model, class_labels)

    elif args.folder:
        # Traiter un dossier d'images
        folder = Path(args.folder)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        for img_path in folder.iterdir():
            if img_path.suffix.lower() in image_extensions:
                print(f"\n{'='*50}")
                print(f"📸 Traitement: {img_path}")
                results, _ = detect_and_classify(str(img_path), yolo_model, mobilenet_model, class_labels)
                format_results(results)
    else:
        # Mode interactif
        print("\n" + "="*50)
        print("🌱 PIPELINE YOLO + MOBILENET")
        print("   Détection et Classification de Maladies de Plantes")
        print("="*50)
        print("\nOptions disponibles:")
        print("  1. Analyser une image")
        print("  2. Analyser une vidéo")
        print("  3. Utiliser la webcam")
        print("  4. Analyser un dossier")
        print("  5. Quitter")

        while True:
            choice = input("\nVotre choix (1-5): ").strip()

            if choice == '1':
                image_path = input("Chemin de l'image: ").strip()
                if os.path.exists(image_path):
                    results, _ = detect_and_classify(image_path, yolo_model, mobilenet_model, class_labels)
                    format_results(results)
                else:
                    print("❌ Fichier non trouvé")

            elif choice == '2':
                video_path = input("Chemin de la vidéo: ").strip()
                if os.path.exists(video_path):
                    process_video(video_path, yolo_model, mobilenet_model, class_labels)
                else:
                    print("❌ Fichier non trouvé")

            elif choice == '3':
                process_webcam(yolo_model, mobilenet_model, class_labels)

            elif choice == '4':
                folder_path = input("Chemin du dossier: ").strip()
                if os.path.isdir(folder_path):
                    folder = Path(folder_path)
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                    for img_path in folder.iterdir():
                        if img_path.suffix.lower() in image_extensions:
                            results, _ = detect_and_classify(str(img_path), yolo_model, mobilenet_model, class_labels)
                            format_results(results)
                else:
                    print("❌ Dossier non trouvé")

            elif choice == '5':
                print("👋 Au revoir!")
                break
            else:
                print("❌ Choix invalide")


if __name__ == "__main__":
    main()

