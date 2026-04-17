from flask import Flask, redirect, url_for, request
from flask import render_template
app = Flask(__name__, template_folder='templates')

import pickle


filename = 'ILANO_90VALIDATION_finalized_model_HCI_no_user_7clusters.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
print("umabot dito")

@app.route('/')
def index():
    return render_template('ILANO_GetData.html')

@app.route('/nakuha/<val>')
def nakuha(val):
    if val == '[\'cluster1\']':
        desc='Cluster 1: The "Right-Handed Male Static Slider User" — Early to mid-career adult males, right-handed, no glasses, using static sliders. Mean Rating of 3.12'
    elif val == '[\'cluster2\']':
        desc='Cluster 2: The "Left-Handed Young Adult Male See-Through Dial User" — Young adult males, left-handed, using see-through dials. Mean Rating of 4.32'
    elif val == '[\'cluster3\']':
        desc='Cluster 3: The "Female Glasses-Wearing See-Through Button User" — Mid-career females, right-handed, wearing glasses, using see-through buttons. Mean Rating of 2.76'
    elif val == '[\'cluster4\']':
        desc='Cluster 4: The "Right-Handed Young Adult Female Static Button User" — Young adult females, right-handed, no glasses, using static buttons. Mean Rating of 3.37'
    elif val == '[\'cluster5\']':
        desc='Cluster 5: The "Middle-Aged Female Dynamic Slider User" — Middle-aged females, right-handed, no glasses, using dynamic sliders. Mean Rating of 3.77.'
    elif val == '[\'cluster6\']':
        desc='Cluster 6: The "Left-Handed Male Glasses-Wearing See-Through Slider User" — Mid-career left-handed males with glasses, using see-through sliders. Mean Rating of 3.80.'
    elif val == '[\'cluster7\']':
        desc='Cluster 7: The "Older Adult Male Dynamic Button User" — Older males, right-handed, no glasses, using dynamic buttons. Mean Rating of 2.66.'
    else:
        desc='Cluster not found.'
    return render_template('ILANO_ModelResults.html', predicted_cluster=val, description=desc)

@app.route('/test', methods=['POST', 'GET'])
def getData():
    if request.method == 'POST':
        object = request.form['object']
        age = request.form['age']
        view = request.form['view']
        rating = request.form['rating']
        gender = request.form['gender']
        glasses = request.form['glasses']
        handedness = request.form['handedness']
        resulta = loaded_model.predict([[int(object), int(age), int(view), int(rating), int(gender), int(glasses), int(handedness)]])
        return redirect(url_for('nakuha', val=resulta))

if __name__ == '__main__':
    app.run(host='192.168.1.8', port='80', debug=True)
