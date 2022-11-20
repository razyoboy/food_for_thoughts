import os

from flask import Flask, request
from flask_restful import Resource, Api

from A2 import predict_result

from waitress import serve

app = Flask(__name__)
api = Api(app)

class CheckHealth(Resource):

    def check_status(self):
        status = {
            "status": 200,
            "text": "Up and Running!"
        }

        return status

    def get(self):
        status = self.check_status()

        return status

api.add_resource(CheckHealth,
    '/api/'
)

class FoodPredict(Resource):

    def post(self):
        try:
            input_image = request.files['image-file']

            confidence, label = predict_result(input_image)
            confidence_normalized = confidence * 100

            results = {
                "status": 200,
                "prediction_results": {
                    "food_type": label.upper(),
                    "confidence": f"{confidence_normalized:.2f}%"
                }
            }
        
        except Exception as e:
            results = {
                "status": 500,
                "text": e
            }
        
        return results

api.add_resource(FoodPredict,
    '/api/predict-food/'    
)

if __name__ == "__main__":
    serve(app, port=int(os.environ.get("PORT", 8080)))
