In this project, we built a complete machine learning pipeline to predict Amazon Sales Amount Prediction using various regression algorithms. We started by:
• Loading and preprocessing the dataset with careful treatment of missing values, scaling, and encoding using a custom pipeline.
• Stratified splitting was used to maintain Quantity_category distribution between train and test sets.
• We trained and evaluated multiple algorithms including:
  • Decision Tree Regressor
  • Random Forest Regressor
  • Linear Regression
Through cross-validation, we found that Random Forest performed the best,offering the lowest RMSE and most stable results.
Finally, we built a script that:
• Trains the Random Forest model and saves it using joblib .
• Uses an if-else logic to skip retraining if the model exists.
• Applies the trained model to new data (input.csv ) to predict TotalAmount, storing results in output.csv .
This pipeline ensures that predictions are accurate, efficient, and ready for production deployment
