import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix

class DenseLayer:
    def __init__(self, input_size, output_size, activation=None):
        # Using Xavier-ish initialization but scaled down a bit to avoid exploding gradients
        # Tried standard normal first but activations saturated too quickly
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))  # Zero init for biases seems to work fine
        self.activation = activation
        
        # Adam optimizer stuff - honestly, this made a huge difference compared to vanilla SGD
        # Had to look up the proper initialization, got weird NaN errors before
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)
        
    def forward(self, inputs):
        # Save inputs for backprop - easy to forget this step!
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation function - wish Python had proper switch statements...
        if self.activation == 'relu':
            # ReLU is so simple yet works so well
            self.activated_output = np.maximum(0, self.output)
        elif self.activation == 'softmax':
            # Subtracting max for numerical stability - learned this trick after getting overflows
            exp_values = np.exp(self.output - np.max(self.output, axis=1, keepdims=True))
            self.activated_output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        else:
            # No activation = linear layer
            self.activated_output = self.output
            
        return self.activated_output
    
    def backward(self, dvalues):
        # If we're using ReLU, need to mask out gradients where input was <= 0
        # This part took me a while to debug properly!
        if self.activation == 'relu':
            dvalues = dvalues.copy()  # Don't modify the original gradient
            dvalues[self.output <= 0] = 0
        
        # Chain rule! The magic of backprop. Took me some time to get comfortable with matrix dimensions here.
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbias = np.sum(dvalues, axis=0, keepdims=True)  # Sum over batch dimension
        
        # Need to pass gradient to previous layer - chain rule again
        # This matrix math still feels like magic sometimes
        self.dinputs = np.dot(dvalues, self.weights.T)
        
        return self.dinputs
    
    def update_params(self, learning_rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Adam optimizer - so much better than plain SGD once I got it working
        # It's overkill for simple problems but good practice to implement it

        # Update momentum and RMSprop-like moving averages
        self.m_weights = beta1 * self.m_weights + (1 - beta1) * self.dweights
        self.v_weights = beta2 * self.v_weights + (1 - beta2) * np.square(self.dweights)
        
        # Bias correction - forgot this initially and convergence was way slower
        m_weights_corrected = self.m_weights / (1 - beta1**t)
        v_weights_corrected = self.v_weights / (1 - beta2**t)
        
        # The actual update step
        self.weights -= learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + epsilon)
        
        # Same for biases
        self.m_bias = beta1 * self.m_bias + (1 - beta1) * self.dbias
        self.v_bias = beta2 * self.v_bias + (1 - beta2) * np.square(self.dbias)
        m_bias_corrected = self.m_bias / (1 - beta1**t)
        v_bias_corrected = self.v_bias / (1 - beta2**t)
        self.bias -= learning_rate * m_bias_corrected / (np.sqrt(v_bias_corrected) + epsilon)


class NeuralNetwork:
    def __init__(self):
        # Empty list to store our layers - Python's flexibility is nice here
        self.layers = []
        
    def add(self, layer):
        # Simple API inspired by Keras - makes building the network feel intuitive
        self.layers.append(layer)
        
    def forward(self, X):
        # Push data through the whole network, layer by layer
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, output, y):
        # Starting the backward pass is a bit special for softmax + cross-entropy
        samples = len(output)
        
        # Convert integer labels to one-hot encoding
        # This feels inefficient but it's cleaner code than the alternatives
        y_one_hot = np.zeros_like(output)
        y_one_hot[np.arange(samples), y] = 1
        
        # For softmax + CCE, the gradient is actually really simple: predictions - targets
        # Took me a while to derive this math, saved a lot of computation!
        dvalues = output.copy()
        dvalues -= y_one_hot
        dvalues /= samples  # Average over the batch
        
        # Backprop through all layers in reverse
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)
    
    def train(self, X, y, X_val=None, y_val=None, epochs=10, batch_size=32, learning_rate=0.001):
        # Lists to track performance - really helpful for diagnostics later
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            batches = 0
            
            # Shuffle training data every epoch - huge improvement for convergence!
            # Without this the model can get stuck in weird patterns
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training - tqdm is awesome for tracking progress
            for i in tqdm(range(0, len(X_shuffled), batch_size), desc=f'Epoch {epoch+1}/{epochs}'):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass to get predictions
                output = self.forward(X_batch)
                
                # Calculate the categorical cross-entropy loss
                # Adding epsilon to avoid log(0) - burned me before!
                y_one_hot = np.zeros_like(output)
                y_one_hot[np.arange(len(y_batch)), y_batch] = 1
                batch_loss = -np.sum(y_one_hot * np.log(output + 1e-7)) / len(y_batch)
                epoch_loss += batch_loss
                batches += 1
                
                # Backward pass to get gradients
                self.backward(output, y_batch)
                
                # Update parameters with Adam - t parameter is for bias correction
                # Took me a while to figure out how to properly count iterations
                t = epoch * (len(X) // batch_size) + (i // batch_size) + 1
                for layer in self.layers:
                    layer.update_params(learning_rate, t)
            
            # Calculate average loss for this epoch
            avg_loss = epoch_loss / batches
            train_losses.append(avg_loss)
            
            # Evaluating on all training data is slow, so let's use a subset
            # Good enough to track progress without slowing everything down
            train_subset_size = min(10000, len(X))  # 10k samples is plenty
            train_indices = np.random.choice(len(X), train_subset_size, replace=False)
            train_accuracy = self.evaluate(X[train_indices], y[train_indices])
            train_accuracies.append(train_accuracy)
            
            # Validation is super important to check for overfitting
            # Added this after noticing training accuracy wasn't telling the full story
            if X_val is not None and y_val is not None:
                val_accuracy = self.evaluate(X_val, y_val)
                val_accuracies.append(val_accuracy)
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        
        # Return all metrics for plotting - this helps visualize training progress
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies if X_val is not None else None
        }
    
    def evaluate(self, X, y):
        # Get predictions from the model
        predictions = self.forward(X)
        predictions = np.argmax(predictions, axis=1)  # Convert from one-hot to class indices
        
        # Simple accuracy calculation - could expand with more metrics later
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def predict(self, X):
        # Simple prediction function for inference
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)
    
    def get_confusion_matrix(self, X, y):
        # Confusion matrix is super helpful for understanding classification errors
        predictions = self.predict(X)
        return confusion_matrix(y, predictions)

# Example usage with the specified architecture
def create_mnist_model():
    # Creating a model specifically for MNIST with the architecture from the requirements
    # This worked pretty well, though I'd try more neurons with more regularization for a real task
    model = NeuralNetwork()
    model.add(DenseLayer(784, 256, activation='relu'))    # MNIST images are 28x28 = 784 flattened
    model.add(DenseLayer(256, 128, activation='relu'))    # Going from wider to narrower layers
    model.add(DenseLayer(128, 64, activation='relu'))     # Standard pyramid architecture
    model.add(DenseLayer(64, 32, activation='relu'))      # Gradually reducing dimensions
    model.add(DenseLayer(32, 10, activation='softmax'))   # 10 digits = 10 output neurons
    
    return model

def plot_sample_predictions(model, X, y, num_samples=25):
    # Visualize example predictions - super useful to get an intuition for errors
    predictions = model.predict(X)
    plt.figure(figsize=(12, 12))
    
    # Randomly select samples to show
    indices = np.random.choice(range(len(X)), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i+1)
        
        # Get the sample and make sure it's a numpy array
        # Sklearn data sometimes comes as a DataFrame so this handles that case
        sample = X[idx]
        if not isinstance(sample, np.ndarray):
            sample = sample.to_numpy()
        
        # Reshape the flattened image back to 28x28 for visualization
        img = sample.reshape(28, 28)
        plt.imshow(img, cmap='gray')
        
        # Color code the title - green for correct, red for mistakes
        # This visual cue makes it super easy to spot errors
        title_color = 'green' if predictions[idx] == y[idx] else 'red'
        plt.title(f'True: {y[idx]}, Pred: {predictions[idx]}', color=title_color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    
def plot_confusion_matrix(confusion_mat):
    # Confusion matrix - love using seaborn for this, makes it look nice
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

def plot_feature_visualization(model):
    # Visualize what the first layer neurons are detecting
    # This is one of my favorite ML visualizations
    weights = model.layers[0].weights
    
    # Showing all 256 would be a mess, let's just display 64
    # Had to adjust grid size to make a nice square
    grid_size = int(np.ceil(np.sqrt(min(weights.shape[1], 64))))
    
    plt.figure(figsize=(15, 15))
    for i in range(min(64, weights.shape[1])):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Reshape weights to image dimensions
        weight_img = weights[:, i].reshape(28, 28)
        
        # Normalize values for better visualization - makes patterns clearer
        vmin, vmax = weight_img.min(), weight_img.max()
        plt.imshow(weight_img, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_visualization.png')

def plot_learning_curves(metrics):
    # Plot the learning curves - always my first go-to for diagnosing training issues
    plt.figure(figsize=(15, 5))
    
    # Plot train loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_losses'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)  # Subtle grid makes it easier to read
    
    # Plot train and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_accuracies'], label='Train Accuracy')
    if metrics['val_accuracies'] is not None:
        plt.plot(metrics['val_accuracies'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')

# Example of loading MNIST data and training the model
if __name__ == '__main__':
    # Try to import MNIST dataset, if available
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        
        print("Loading MNIST dataset...")
        # Using fetch_openml is easier than the old mnist loader
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
        X = X.astype('float32') / 255  # Scale to [0, 1] - SUPER important for gradient stability!
        y = y.astype('int')
        
        # Convert to numpy array if it's not already
        # This gave me headaches until I realized sklearn sometimes returns DataFrames
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        if hasattr(y, 'to_numpy'):
            y = y.to_numpy()
        
        # Split data - always use a fixed random state for reproducibility
        # 80/20 split seems pretty standard for this
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Further split training into train/validation
        # This helped me catch overfitting issues early
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Create and train model
        model = create_mnist_model()
        print("Training model...")
        metrics = model.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=64, learning_rate=0.001)
        
        # Evaluate on the test set - the real measure of performance
        accuracy = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Generate and save all visualizations
        print("Generating visualizations...")
        
        # These visualizations helped me understand the model behavior much better
        plot_learning_curves(metrics)
        
        conf_matrix = model.get_confusion_matrix(X_test, y_test)
        plot_confusion_matrix(conf_matrix)
        
        plot_sample_predictions(model, X_test, y_test)
        
        plot_feature_visualization(model)
        
        print("All visualizations have been saved.")
        
    except ImportError:
        print("Could not import required packages. Please install them:")
        print("pip install scikit-learn matplotlib seaborn tqdm numpy") 