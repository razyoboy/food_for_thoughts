# Food for Thoughts

![Frame 68](https://user-images.githubusercontent.com/78082583/205480192-a7bc2578-abf2-457d-821b-e2addd95cd2a.png)

A Tensorflow-based ML model for food classification and detection, based on the Kaggle dataset `food-101` and `thfood-50` - utilizing the `InceptionV3` Convolutional Neural Network (CNN) model.

## Usage (via API)
This can be done locally, or via a Cloud serverless provider of your choice. In our case, we used Google Cloud Platform as the main Cloud provider.

### Deployment via Google Cloud Run (CLI-method)

1. Install `gcloud` CLI and follow the instructions as shown [here](https://cloud.google.com/sdk/docs/install-sdk)
2. Clone this repository and navigate to its root directory
```
git clone https://github.com/razyoboy/food_for_thoughts/
cd food_for_thoughts
```
3. Deploy to your project by using the following command (more options can be found [here](https://cloud.google.com/run/docs/deploying))
```
gcloud run deploy
```

### Deployment via Local PC
1. Clone this repository and navigate to the root directory
```
git clone https://github.com/razyoboy/food_for_thoughts/
cd food_for_thoughts
```
2. Ensure all requirements are met
```
pip install -r requirements.txt
```
3. Run the local server
```
python -u "src/main.py"
```

### API Endpoints and Interaction

Currently, `food_for_thoughts` supports two endpoints:
#### `GET` Check Health
  * `<base_url>/api/`
  
  Returns 
```
{
    "status": 200,
    "text": "Up and Running!"
}
```
  
#### `POST` Predict Food
  * `<base_url>/api/predict-food/`
  * `form-data` body with `image-file` as the key
  * Attach image via the `image-file` key
  
  Returns
  ```
  {
   "status": 200,
   "prediction_results": {
       "food_type": <food_result>,
       "confidence": <percentage>
   }
}
```

<sub>_DISCLAIMER: This project is part of EGBI443 Image Processing in Medicine class project, and is not intended to be a production ready implementation for food prediction via API calls_
