from flask import Flask, request, render_template, jsonify
import pandas as pd
from config.paths_config import MODEL_PATH, MODEL_DIR
from src.logger import get_logger
from rapidfuzz.process import extract  
import os
import pickle


app = Flask(__name__, template_folder='templates', static_folder='static')
logger = get_logger(__name__)

try:
    with open(MODEL_PATH,'rb') as inp:
        model                   = pickle.load(inp)
        Movie_titles            = pickle.load(inp)
        
    logger.info("Model loaded successfully from %s", MODEL_PATH)
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/suggest")
def suggest():
    #function to get suggestions as we enter the movie name
    my_search_string = request.args.get('q', '')
    filtered_titles = extract(my_search_string,Movie_titles,limit=10)
    suggestions = list(set([tup[0] for tup in filtered_titles]))
    return jsonify(suggestions)


@app.route('/predict', methods=['GET'])
def predict():
    try:
        data = request.args.get('movie')
        # request.form.to_dict()
        logger.info(f"Received form data: {data}")
        prediction = model.similar_items(data, n_items = 5)
        logger.info(f"Prediction: {prediction}")
        # Return prediction
        return jsonify({
            'prediction': prediction.tolist()
        })
    except KeyError as e:
        logger.error(f"Missing field: {e}")
        return jsonify({'error': f"Missing required field: {e}"}), 400
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)