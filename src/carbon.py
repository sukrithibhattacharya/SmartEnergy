import numpy as np

EMISSION_FACTOR = 0.4  # kg CO₂ per kWh

def calculate_carbon_savings(predicted_load, actual_load):
    """
    Calculate CO₂ savings from using AI predictions
    predicted_load: array of predicted loads
    actual_load: array of actual loads
    """
    energy_saved = np.sum(np.abs(predicted_load - actual_load))
    carbon_saved = energy_saved * EMISSION_FACTOR
    trees_equivalent = carbon_saved / 25  # 1 tree absorbs ~25kg CO₂/year
    money_saved = carbon_saved * 0.10  # ~$0.10 per kg CO₂
    
    return {
        'energy_saved_kwh': energy_saved,
        'carbon_saved_kg': carbon_saved,
        'trees_equivalent': trees_equivalent,
        'money_saved_usd': money_saved
    }