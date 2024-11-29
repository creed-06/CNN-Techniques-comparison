import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalMaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def page2():

    # Keep the existing preprocessing and model creation functions unchanged
    # [Previous preprocess_data_for_cnn, create_1d_cnn_model, create_2d_cnn_model functions remain the same]

    def display_data_overview(data):
        """Display data overview with statistics and visualizations"""
        st.write("#### 📊 Dataset Overview")
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            st.metric("Positive Cases", len(data[data['target'] == 1]))
        with col3:
            st.metric("Negative Cases", len(data[data['target'] == 0]))
        
        # Show data sample in an expandable section
        with st.expander("View Dataset Sample"):
            st.dataframe(
                data.head(10).style.background_gradient(cmap='Blues'),
                use_container_width=True
            )
            
            # Display feature descriptions
            st.markdown("""
            **Feature Descriptions:**
            - **age**: Age in years
            - **sex**: Gender (1 = male; 0 = female)
            - **cp**: Chest pain type
            - **trestbps**: Resting blood pressure
            - **chol**: Serum cholesterol in mg/dl
            - **target**: Heart disease diagnosis (1 = present; 0 = absent)
            """)

    def display_model_comparison(y_test, y_pred_1d, y_pred_2d, history_1d, history_2d):
        """Display comprehensive model comparison"""
        st.write("#### 🔍 Model Performance Comparison")
        
        # Create metrics table
        metrics_1d = calculate_metrics(y_test, y_pred_1d)
        metrics_2d = calculate_metrics(y_test, y_pred_2d)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            '1D CNN': metrics_1d,
            '2D CNN': metrics_2d
        })
        
        # Style the metrics table
        st.dataframe(
            metrics_df.style.background_gradient(subset=['1D CNN', '2D CNN'], cmap='Blues'),
            use_container_width=True
        )
        
        # Add interpretation guide
        with st.expander("📖 How to Interpret These Metrics"):
            st.markdown("""
            **Key Metrics Explained:**
            1. **Accuracy**: Overall correct predictions percentage
                - Good: > 85%
                - Acceptable: 75-85%
                - Needs Improvement: < 75%
            
            2. **Precision**: Accuracy of positive predictions
                - High precision means fewer false positives
                - Critical for avoiding unnecessary medical interventions
            
            3. **Recall**: Ability to detect actual positive cases
                - High recall means fewer missed heart disease cases
                - Critical for patient safety
            
            4. **F1-Score**: Balance between precision and recall
                - Best when precision and recall are both high
                - Good overall performance indicator
            
            5. **AUC-ROC**: Model's discriminative ability
                - Excellent: > 0.90
                - Good: 0.80-0.90
                - Fair: 0.70-0.80
                - Poor: < 0.70
            """)

    def calculate_metrics(y_true, y_pred):
        """Calculate comprehensive metrics for model evaluation"""
        y_true_class = np.argmax(y_true, axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)
        
        # Calculate basic metrics
        accuracy = np.mean(y_true_class == y_pred_class)
        precision = np.sum((y_true_class == 1) & (y_pred_class == 1)) / np.sum(y_pred_class == 1)
        recall = np.sum((y_true_class == 1) & (y_pred_class == 1)) / np.sum(y_true_class == 1)
        f1 = 2 * (precision * recall) / (precision + recall)
        auc = roc_auc_score(y_true[:, 1], y_pred[:, 1])
        
        return [f"{accuracy:.3f}", f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}", f"{auc:.3f}"]

    def preprocess_data_for_cnn(data, model_type='1d'):
        """
        Comprehensive data preprocessing for CNN models
        
        Args:
            data (pd.DataFrame): Input dataframe
            model_type (str): '1d' or '2d' CNN model
        
        Returns:
            Preprocessed features and categorical target
        """
        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # One-hot encode categorical variables
        categorical_features = ['cp', 'restecg', 'slope', 'thal']
        # Handle missing categorical features
        available_cat_features = [col for col in categorical_features if col in X.columns]
        
        if available_cat_features:
            onehot = OneHotEncoder(sparse_output=False)
            encoded_features = onehot.fit_transform(X[available_cat_features])
            
            # Create encoded dataframe
            encoded_df = pd.DataFrame(
                encoded_features, 
                columns=onehot.get_feature_names_out(available_cat_features)
            )
            
            # Combine encoded and original features
            X = pd.concat([
                X.drop(available_cat_features, axis=1).reset_index(drop=True), 
                encoded_df
            ], axis=1)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert target to categorical
        y_cat = to_categorical(y)
        
        # Reshape based on model type
        if model_type == '1d':
            # Reshape for 1D CNN: (samples, features, 1)
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        elif model_type == '2d':
            # Ensure square-ish grid
            num_features = X_scaled.shape[1]
            height = int(np.ceil(np.sqrt(num_features)))
            width = int(np.ceil(num_features / height))
            
            # Pad if necessary
            padded_features = np.zeros((X_scaled.shape[0], height * width))
            padded_features[:, :num_features] = X_scaled
            
            # Reshape for 2D CNN
            X_reshaped = padded_features.reshape(
                X_scaled.shape[0], height, width, 1
            )
        else:
            raise ValueError("model_type must be '1d' or '2d'")
        
        return X_reshaped, y_cat, scaler

    def create_1d_cnn_model(input_shape, num_classes):
        """Create 1D CNN model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            Dropout(0.3),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def create_2d_cnn_model(input_shape, num_classes):
        """Create 2D CNN model"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def plot_training_history(history_1d, history_2d):
        """Plot training history for both models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy Plot
        ax1.plot(history_1d.history['accuracy'], label='1D CNN Train')
        ax1.plot(history_1d.history['val_accuracy'], label='1D CNN Validation')
        ax1.plot(history_2d.history['accuracy'], label='2D CNN Train')
        ax1.plot(history_2d.history['val_accuracy'], label='2D CNN Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        # Loss Plot
        ax2.plot(history_1d.history['loss'], label='1D CNN Train')
        ax2.plot(history_1d.history['val_loss'], label='1D CNN Validation')
        ax2.plot(history_2d.history['loss'], label='2D CNN Train')
        ax2.plot(history_2d.history['val_loss'], label='2D CNN Validation')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        plt.tight_layout()
        return fig

    def plot_roc_curves(y_test, y_pred_1d, y_pred_2d):
        """Plot ROC curves for both models"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # ROC for 1D CNN
        fpr_1d, tpr_1d, _ = roc_curve(y_test[:, 1], y_pred_1d[:, 1])
        roc_auc_1d = roc_auc_score(y_test[:, 1], y_pred_1d[:, 1])
        ax.plot(fpr_1d, tpr_1d, label=f'1D CNN (AUC = {roc_auc_1d:.2f})')
        
        # ROC for 2D CNN
        fpr_2d, tpr_2d, _ = roc_curve(y_test[:, 1], y_pred_2d[:, 1])
        roc_auc_2d = roc_auc_score(y_test[:, 1], y_pred_2d[:, 1])
        ax.plot(fpr_2d, tpr_2d, label=f'2D CNN (AUC = {roc_auc_2d:.2f})')
        
        ax.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        return fig

    def plot_confusion_matrices(y_test, y_pred_1d, y_pred_2d):
        """Plot confusion matrices for both models"""
        # Convert predictions to class labels
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_1d_labels = np.argmax(y_pred_1d, axis=1)
        y_pred_2d_labels = np.argmax(y_pred_2d, axis=1)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 1D CNN Confusion Matrix
        cm_1d = confusion_matrix(y_test_labels, y_pred_1d_labels)
        sns.heatmap(cm_1d, annot=True, fmt='d', ax=ax1, cmap='Blues')
        ax1.set_title('1D CNN Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 2D CNN Confusion Matrix
        cm_2d = confusion_matrix(y_test_labels, y_pred_2d_labels)
        sns.heatmap(cm_2d, annot=True, fmt='d', ax=ax2, cmap='Blues')
        ax2.set_title('2D CNN Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        return fig

    def main():
        # Page Configuration

        
        # Header
        st.title("🫀 Heart Disease Detection with CNN Models")
        st.markdown("""
        This application compares 1D and 2D CNN models for heart disease detection using clinical data.
        The analysis provides insights into model performance and helps identify the most effective approach.
        """)
        
        # Load data
        data = pd.read_csv('C:/VS code/IDP/Data/heart.csv')  # Adjust path as needed
        
        # Display data overview
        display_data_overview(data)
        
        # Model Configuration Section
        st.write("### ⚙️ Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Training size slider
            train_size = st.slider(
                "Training Data Size",
                min_value=0.5,
                max_value=0.9,
                value=0.8,
                step=0.05,
                help="Proportion of data to use for training (remaining will be used for testing)"
            )
            
        with col2:
            # Epochs slider
            epochs = st.slider(
                "Number of Epochs",
                min_value=10,
                max_value=100,
                value=50,
                step=10,
                help="Number of training iterations"
            )
        
        with col3:
            # Early stopping patience
            patience = st.slider(
                "Early Stopping Patience",
                min_value=5,
                max_value=20,
                value=10,
                step=1,
                help="Number of epochs to wait before stopping if no improvement"
            )
        
        # Add a start training button
        if st.button("🚀 Start Training", type="primary"):
            # Model Training Section
            st.write("### 🏃‍♂️ Training Models")
            
            # Progress placeholder
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Prepare data
            X_1d, y_1d, _ = preprocess_data_for_cnn(data, model_type='1d')
            X_2d, y_2d, _ = preprocess_data_for_cnn(data, model_type='2d')
            
            # Calculate test size from train size
            test_size = 1 - train_size
            
            # Split data with user-defined ratio
            X_train_1d, X_test_1d, y_train_1d, y_test_1d = train_test_split(
                X_1d, y_1d, test_size=test_size, random_state=42
            )
            X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
                X_2d, y_2d, test_size=test_size, random_state=42
            )
            
            # Display split information
            st.info(f"""
            Data Split Information:
            - Training samples: {len(X_train_1d)} ({train_size*100:.1f}%)
            - Testing samples: {len(X_test_1d)} ({test_size*100:.1f}%)
            """)
            
            # Train models with progress updates
            status_text.text('Training 1D CNN...')
            progress_bar.progress(25)
            
            model_1d = create_1d_cnn_model(
                input_shape=(X_train_1d.shape[1], 1),
                num_classes=y_1d.shape[1]
            )
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            )
            
            history_1d = model_1d.fit(
                X_train_1d, y_train_1d,
                validation_split=0.2,
                epochs=epochs,
                callbacks=[early_stopping],
                verbose=0
            )
            
            progress_bar.progress(50)
            status_text.text('Training 2D CNN...')
            
            model_2d = create_2d_cnn_model(
                input_shape=X_train_2d.shape[1:],
                num_classes=y_2d.shape[1]
            )
            
            history_2d = model_2d.fit(
                X_train_2d, y_train_2d,
                validation_split=0.2,
                epochs=epochs,
                callbacks=[early_stopping],
                verbose=0
            )
            
            progress_bar.progress(100)
            status_text.text('Training completed!')
            
            # Generate predictions
            y_pred_1d = model_1d.predict(X_test_1d)
            y_pred_2d = model_2d.predict(X_test_2d)
            
            # Training Summary
            st.success(f"""
            Training Completed Successfully:
            - 1D CNN stopped after {len(history_1d.history['loss'])} epochs
            - 2D CNN stopped after {len(history_2d.history['loss'])} epochs
            - Early stopping patience: {patience} epochs
            """)
            
            # Display model comparison
            display_model_comparison(y_test_1d, y_pred_1d, y_pred_2d, history_1d, history_2d)
            
            # Visualization Section
            st.write("### 📈 Performance Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Training History")
                fig_history = plot_training_history(history_1d, history_2d)
                st.pyplot(fig_history)
            
            with col2:
                st.write("#### ROC Curves")
                fig_roc = plot_roc_curves(y_test_1d, y_pred_1d, y_pred_2d)
                st.pyplot(fig_roc)
            
            # Confusion Matrices
            st.write("#### Confusion Matrices")
            fig_cm = plot_confusion_matrices(y_test_1d, y_pred_1d, y_pred_2d)
            st.pyplot(fig_cm)
            
            # Training Impact Analysis
            st.write("### 🔍 Training Configuration Impact")
            st.markdown(f"""
            **Impact of Selected Parameters:**
            
            1. **Training Data Size ({train_size*100:.1f}%)**:
            - Larger training size ({train_size*100:.1f}%) provides more data for model learning
            - Smaller test size ({test_size*100:.1f}%) means less data for evaluation
            - Current split appears to be {'optimal' if 0.7 <= train_size <= 0.8 else 'sub-optimal'}
            
            2. **Number of Epochs ({epochs})**:
            - 1D CNN used {len(history_1d.history['loss'])} epochs before early stopping
            - 2D CNN used {len(history_2d.history['loss'])} epochs before early stopping
            - Early stopping patience of {patience} epochs helped prevent overfitting
            
            3. **Model Convergence**:
            - 1D CNN final validation loss: {history_1d.history['val_loss'][-1]:.4f}
            - 2D CNN final validation loss: {history_2d.history['val_loss'][-1]:.4f}
            - {'Both models converged well' if history_1d.history['val_loss'][-1] < 0.5 and history_2d.history['val_loss'][-1] < 0.5 else 'Models might benefit from additional tuning'}
            """)
            
            # Final Conclusions
            st.write("### 🎯 Final Analysis")
            metrics_1d = calculate_metrics(y_test_1d, y_pred_1d)
            metrics_2d = calculate_metrics(y_test_2d, y_pred_2d)
            
            conclusion = f"""
            Based on the analysis with {train_size*100:.1f}% training data:
            
            1. **Overall Performance**:
            - 1D CNN achieved {metrics_1d[0]} accuracy with {metrics_1d[4]} AUC-ROC
            - 2D CNN achieved {metrics_2d[0]} accuracy with {metrics_2d[4]} AUC-ROC
            
            2. **Model Comparison**:
            - {'1D CNN' if float(metrics_1d[0]) > float(metrics_2d[0]) else '2D CNN'} shows better overall accuracy
            - {'1D CNN' if float(metrics_1d[4]) > float(metrics_2d[4]) else '2D CNN'} demonstrates superior discriminative ability
            
            3. **Clinical Implications**:
            - Precision: {'High' if float(metrics_1d[1]) > 0.85 else 'Moderate'} confidence in positive predictions
            - Recall: {'Strong' if float(metrics_1d[2]) > 0.85 else 'Moderate'} ability to identify actual heart disease cases
            
            4. **Training Configuration Impact**:
            - Selected training size appears to be {'optimal' if 0.7 <= train_size <= 0.8 else 'sub-optimal'} for this dataset
            - Early stopping helped prevent overfitting while maintaining performance
            """
            
            st.markdown(conclusion)
        
        else:
            st.info("👆 Configure the training parameters and click 'Start Training' to begin the analysis.")
        # Call main() at the end of page2()
    main()
