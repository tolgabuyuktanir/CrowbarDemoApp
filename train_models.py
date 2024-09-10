def train_model(file):
    # Here, you can add your code to train the model using the provided parameters.
    # For demonstration purposes, we'll just return the collected parameters.
    print(file.name)
    return f"""
    File: {file.name if file else 'No file uploaded'}
    """