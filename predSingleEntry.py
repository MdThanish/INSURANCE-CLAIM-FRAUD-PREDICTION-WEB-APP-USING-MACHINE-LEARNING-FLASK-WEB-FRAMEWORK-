import pandas as pd
import numpy as np
from flask import request
from Data_Preprocessor.Data_preprocessor import preProcessing
from fileOperations.fileMethods import fileMethods
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import plot_confusion_matrix

class PredFromRec:
    def get_values(self):
        try:
            policy_deductible = int(request.form['policy_deductable'])
            policy_annual_premium = float(request.form['policy_annual_premium'])
            umbrella_limit = int(request.form['umbrella_limit'])
            incident_type = request.form['incident_type']
            collision_type = request.form['collision_type']
            incident_severity = request.form['incident_severity']
            authorities_contacted = request.form['authorities_contacted']
            number_of_vehicles_involved = int(request.form['number_of_vehicles_involved'])
            witnesses = int(request.form['witnesses'])
            property_claim = int(request.form['property_claim'])
            insured_hobbies = request.form['insured_hobbies']
            incident_state = request.form['incident_state']

            feature_dict = {
                'policy_deductable': policy_deductible,
                'policy_annual_premium': policy_annual_premium,
                'umbrella_limit': umbrella_limit,
                'incident_type': incident_type,
                'collision_type': collision_type,
                'incident_severity': incident_severity,
                'authorities_contacted': authorities_contacted,
                'number_of_vehicles_involved': number_of_vehicles_involved,
                'witnesses': witnesses,
                'property_claim': property_claim,
                'insured_hobbies': insured_hobbies,
                'incident_state': incident_state,
            }
            return feature_dict
        except Exception as e:
            return f"Error: {str(e)}"

    def pred_from_rec(self, rec):
        filepath = 'goodDataToPred/goodPredData.csv'
        preprocessor = preProcessing()
        data = preprocessor.loadData(filepath)

        features = ['insured_hobbies', 'incident_type', 'collision_type', 'incident_severity',
                    'authorities_contacted', 'incident_state', 'property_damage', 'umbrella_limit',
                    'policy_deductable', 'number_of_vehicles_involved', 'witnesses',
                    'policy_annual_premium', 'property_claim']

        data = preprocessor.removeColumns(data, features)
        data = data.append(rec, ignore_index=True)
        data = preprocessor.removeWhiteSpaces(data)
        data = preprocessor.cleanup(data)
        data = preprocessor.imputeMissingValues(data)
        data = preprocessor.encodeCatcols(data)

        predRec = data.iloc[-1]

        fileops = fileMethods()
        modelName = fileops.findBestModel()
        model = fileops.modelLoader(modelName)
        predRec = predRec.drop('fraud_reported')  # Remove the target variable

        data_pred = model.predict(predRec.values.reshape(1, -1))

        if data_pred[0] == 0:
            return 'No'
        else:
            return 'Yes'

