import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for,jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
from datetime import date
from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from PIL import Image



app = Flask(__name__)
patients = []  # Liste globale des patients détectés
patientspar = []  # Liste globale des patients détectés
total_patients = 0
test_count = 0

patientspar = []  

# Charger les modèles
model_alzheimer = load_model("mon_modele_cnn.keras")
model_parkinson = load_model("mon_modele_cnn_prk.keras")
model_hannddrawn = load_model("mon_modele_cnn_handdraw.keras")
print("Input shape du modèle handdrawn :", model_parkinson.input_shape)



# Paramètres
IMG_SIZE = (150, 150)
def preprocess_image_for_model(img_path, expected_size=(224, 224), flatten=False):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(expected_size)
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, *img_array.shape)

        if flatten:
            img_array = img_array.reshape(1, -1)  # ⚠️ Applatir pour les modèles dense

        return img_array
    except Exception as e:
        raise ValueError(f"Erreur lors du prétraitement de l'image : {str(e)}")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    disease = request.form.get('disease')
    if disease == 'alzheimer':
        return redirect(url_for('alzheimer_page'))
    elif disease == 'parkinson':
        return redirect(url_for('parkinson_page'))
    return "Invalid selection"

@app.route('/alzheimer')
def alzheimer_page():
    today = date.today().isoformat()
    return render_template('alzheimer.html', current_date=today, test_count=test_count, total_patients=total_patients)


    

@app.route('/parkinson')
def parkinson_page():
    today = date.today().isoformat()
    return render_template('parkinson.html', current_date=today, test_count=test_count, total_patients=total_patients)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

from datetime import date
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

# Données globales
patients = []
total_patients = 0
test_count = 0

from flask import request, render_template, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from datetime import date
import os
import numpy as np

@app.route('/predict_alzheimer', methods=['POST'])
def predict_alzheimer():
    global total_patients, test_count, patients

    # Charger le modèle une seule fois si besoin (à optimiser si fait souvent)
    model = load_model("mon_modele_cnn.keras")

    # Récupérer les données du formulaire
    image = request.files['image']
    first_name = request.form.get("first_name")
    last_name = request.form.get("last_name")
    action = request.form.get("action")

    # Sécuriser et sauvegarder l’image
    filename = secure_filename(image.filename)
    image_path = os.path.join("static", filename)
    image.save(image_path)

    # Prétraitement de l’image
    img = load_img(image_path, target_size=(150, 150))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prédiction
    prediction = model.predict(img)[0]
    label = np.argmax(prediction)
    classes = ['MildDemented', 'ModerateDemented', 'NonDemented','VeryMildDemented']
    result = classes[label]

    # Déterminer le statuts
    status = "healthy" if result == "NonDemented" else "sick"

    # Date du test
    today = date.today().isoformat()

    # Mise à jour des stats et enregistrement du patient
    if action == "detect":
        total_patients += 1
        test_count += 1
        patients.append({
            "name": f"{first_name} {last_name}",
            "image": filename,  # ✅ juste le nom
            "status": status,
            "class_name": result,
            "date": today
        })

    # Afficher résultat
    return render_template(
        'alzheimer.html',
        current_date=today,
        test_count=test_count,
        total_patients=total_patients,
        prediction=result,
        image_url=filename,  # ✅ image à afficher dans alzheimer.html
        first_name=first_name,
        last_name=last_name
    )

from flask import request, render_template, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import datetime
from werkzeug.utils import secure_filename

# Charger les modèles une seule fois
model_parkinson = load_model("mon_modele_cnn_prk.keras")
model_handdrawn = load_model("mon_modele_cnn_handdraw.keras")

from flask import request, render_template
import os
from datetime import date
from werkzeug.utils import secure_filename

from PIL import Image
import numpy as np
import os
import datetime
from flask import render_template, request
# Assure-toi que les modèles sont bien chargés en haut :
# model_parkinson, model_hannddrawn

from flask import request, render_template
import os
import numpy as np
from PIL import Image
import datetime

from flask import request, render_template
from PIL import Image
import numpy as np
import os
import datetime

@app.route('/detect_parkinson', methods=['POST'])
def detect_parkinson():
    global patientspar

    test_type = request.form.get('testType')
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')

    if 'file' not in request.files:
        return render_template("parkinson.html", prediction="❌ Aucune image reçue.")

    file = request.files['file']
    if file.filename == '':
        return render_template("parkinson.html", prediction="❌ Aucun fichier sélectionné.")

    # Enregistrer l'image
    save_dir = os.path.join('static', 'uploads')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)
    file.save(save_path)

    try:
        img = Image.open(save_path).convert('RGB')

        if test_type == 'MRI':
            img = img.resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model_parkinson.predict(img_array)
            categories = ['Normal', 'Parkinson']

        elif test_type == 'Drawing':
            img = img.resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model_hannddrawn.predict(img_array)
            categories = ['Healthy', 'Parkinson']

        else:
            return render_template("parkinson.html", prediction="❌ Type de test invalide.")

        result_class = categories[np.argmax(prediction)]

    except Exception as e:
        return render_template("parkinson.html", prediction=f"❌ Erreur lors de la prédiction: {e}")

    # Sauvegarder dans la liste globale
    image_url = f"uploads/{file.filename}"
    status = "sick" if result_class == "Parkinson" else "healthy"
    patientspar.append({
        "name": f"{first_name} {last_name}",
        "image": image_url,
        "status": status,
        "class_name": result_class,
        "date": datetime.date.today().isoformat()
    })

    # Option : rediriger vers liste patients
    return redirect(url_for('patientss_page'))


@app.route('/dashboard')
def dashboard():
    class_counter = {}
    date_counter = {}
    sick = 0
    healthy = 0

    for p in patients:
        # Sécurité : vérifier que la date existe
        date_str = p.get('date')
        if not date_str:
            continue

        # Compter par classe
        class_name = p['class_name']
        class_counter[class_name] = class_counter.get(class_name, 0) + 1

        # Compter par date
        date_counter[date_str] = date_counter.get(date_str, 0) + 1

        # Statut
        if p['status'] == 'sick':
            sick += 1
        else:
            healthy += 1

    # Tri des dates pour affichage chronologique
    sorted_dates = sorted(date_counter.keys())
    sorted_values = [date_counter[d] for d in sorted_dates]

    return render_template('dashboard.html',
                           total=len(patients),
                           sick=sick,
                           healthy=healthy,
                           class_labels=list(class_counter.keys()),
                           class_values=list(class_counter.values()),
                           date_labels=sorted_dates,
                           date_values=sorted_values)



@app.route('/dashboardparkinson')
def dashboardpar():
    class_counter = {}
    date_counter = {}
    sick = 0
    healthy = 0

    for p in patientspar:  # ✅ Utiliser patientspar ici
        date_str = p.get('date')
        if not date_str:
            continue

        class_name = p['class_name']
        class_counter[class_name] = class_counter.get(class_name, 0) + 1
        date_counter[date_str] = date_counter.get(date_str, 0) + 1

        if p['status'] == 'sick':
            sick += 1
        else:
            healthy += 1

    sorted_dates = sorted(date_counter.keys())
    sorted_values = [date_counter[d] for d in sorted_dates]

    return render_template('dashbpar.html',
                           total=len(patientspar),  # ✅ patientspar
                           sick=sick,
                           healthy=healthy,
                           class_labels=list(class_counter.keys()),
                           class_values=list(class_counter.values()),
                           date_labels=sorted_dates,
                           date_values=sorted_values)






@app.route('/patients')
def patients_page():

    return render_template('patient_records.html', patients=patients)
@app.route('/patientss')
def patientss_page():
    return render_template('patient_recordsparkinson.html', patients=patientspar)



@app.route('/delete_patient/<int:index>', methods=['POST'])
def delete_patient(index):
    if 0 <= index < len(patients):
        del patients[index]
    return redirect(url_for('patients_page'))

@app.route('/deletee_patient/<int:index>', methods=['POST'])
def deletee_patient(index):
    if 0 <= index < len(patients):
        del patients[index]
    return redirect(url_for('patientss_page'))



if __name__ == '__main__':
    app.run(debug=True)
