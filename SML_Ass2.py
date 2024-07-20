import numpy as np
import matplotlib.pyplot as plt


def display_sample_images(samples, labels, num_samples=5):
    unique_labels = np.unique(labels)

    for sample_index in range(num_samples):
        for label_index, label_value in enumerate(unique_labels):
            class_indices = np.where(labels == label_value)[0]
            selected_index = np.random.choice(class_indices, 1)[0]

            image = samples[selected_index]

            plt.subplot(num_samples, len(unique_labels), sample_index * len(unique_labels) + label_index + 1)
            plt.imshow(image, cmap='gray')
            plt.axis('off')

    plt.show()


def QDA(x, class_mean, class_cov_inv, class_prior):
    quadratic_term = np.dot(np.transpose(x), np.dot(-0.5 * class_cov_inv, x))
    linear_term = np.dot(np.transpose(np.dot(class_cov_inv, class_mean)), x)
    constant_term = -0.5 * (np.transpose(class_mean).dot(class_cov_inv).dot(class_mean)) + np.log(class_prior)
    result = quadratic_term + linear_term + constant_term
    return result


def calculate_class_statistics(features, labels):
    unique_labels = np.unique(labels)
    class_means = []
    for label in unique_labels:
        class_indices = np.where(labels == label)[0]
        class_mean = np.mean(features[class_indices], axis=0)
        class_means.append(class_mean)

    # class_means = [np.mean(features[labels == label], axis=0) for label in unique_labels]
    class_covariances = [np.cov(features[labels == label], rowvar=False) for label in unique_labels]
    return class_means, class_covariances


def calculate_discriminant_scores(feature, class_means, class_cov_invs, class_priors):
    discriminant_scores = []
    for i in range(len(class_means)):
        mean = class_means[i]
        cov_inv = class_cov_invs[i]
        prior = class_priors[i]
        discriminant_score = QDA(feature, mean, cov_inv, prior)
        discriminant_scores.append(discriminant_score)
    return discriminant_scores


def train_quadratic_discriminant_analysis(features, labels):
    class_means, class_covariances = calculate_class_statistics(features, labels)
    class_cov_invs = [np.linalg.inv(np.array(cov) + np.eye(len(cov)) * 1e-6) for cov in class_covariances]
    class_priors = [(labels == label).sum() / len(labels) for label in np.unique(labels)]
    return class_means, class_cov_invs, class_priors


def predict_quadratic_discriminant_analysis(features, class_means, class_cov_invs, class_priors):
    predicted_classes = []
    for feature in features:
        discriminant_scores = calculate_discriminant_scores(feature, class_means, class_cov_invs, class_priors)
        predicted_class = np.argmax(discriminant_scores)
        predicted_classes.append(predicted_class)
    return np.array(predicted_classes)


def calculate_individual_class_accuracy(predictions, true_labels, class_label):
    class_indices = (true_labels == class_label)
    if np.sum(class_indices) > 0:
        correct_predictions = np.sum(predictions[class_indices] == true_labels[class_indices])
        total_samples = np.sum(class_indices)
        accuracy = correct_predictions / total_samples
    else:
        accuracy = 0.0
    return accuracy


# Display random samples from each class
with np.load('mnist.npz') as data:
    training_images = data["x_train"]
    training_labels = data["y_train"]
    testing_images = data["x_test"]
    testing_labels = data["y_test"]

display_sample_images(training_images, training_labels)


def partOne():
    # Training Quadratic Discriminant Analysis
    class_means, class_cov_invs, class_priors = train_quadratic_discriminant_analysis(
        training_images.reshape(-1, 784),
        training_labels
    )

    # Testing Quadratic Discriminant Analysis
    testing_images_reshaped = testing_images.reshape(-1, 784)
    predicted_classes = predict_quadratic_discriminant_analysis(
        testing_images_reshaped,
        class_means,
        class_cov_invs,
        class_priors
    )

    # Calculate overall accuracy
    correct_predictions = np.sum(predicted_classes == testing_labels)
    total_samples = len(testing_labels)
    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print("Overall Accuracy:", overall_accuracy)

    # Calculate and print class-wise accuracy
    unique_test_labels = np.unique(testing_labels)
    for class_label in unique_test_labels:
        class_accuracy = calculate_individual_class_accuracy(predicted_classes, testing_labels, class_label)
        print(f"Class {class_label} Accuracy: {class_accuracy}")


def getMSE():
    samples_list = []

    for class_label in range(10):
        indices = np.where(training_labels == class_label)[0]

        vectorized = training_images.reshape(training_images.shape[0], -1)
        class_samples = vectorized[indices[:100]]

        # Append the class samples to the list
        samples_list.extend(class_samples)

    # Stack all samples into columns of a matrix
    samples_matrix = np.array(samples_list).T

    # Compute mean of each feature (row-wise mean)
    X_mean = np.mean(samples_matrix, axis=1, keepdims=True)

    # Center the data
    X_centered = samples_matrix - X_mean

    # Calculate the covariance matrix
    covariance_matrix = np.dot(X_centered, X_centered.T) / 999

    # Eigen decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    U = eigenvectors[:, sorted_indices]

    # Project the centered data onto the principal components
    Y = np.dot(U.T, X_centered)

    # Reconstruct the data using the principal components
    X_reconstructed = np.dot(U, Y) + X_mean

    # Calculate Mean Squared Error (MSE)
    MSE = np.sum((X_reconstructed - samples_matrix) ** 2)
    print("MSE:", MSE)
    p_values = [5, 15, 30]


    for p in p_values:

        Up = np.dot(U[:, :p], np.dot(U[:, :p].T, X_centered))


        X_reshaped = (Up + X_mean).reshape(28, 28, -1)

        plt.figure(figsize=(20, 20))
        plt.suptitle(f"p = {p}", fontsize=16, y=0.95)

        for i in range(10):
            indices = np.arange(i * 100, (i + 1) * 100, 20)

            for j, index in enumerate(indices):
                plt.subplot(5, 10, j * 10 + i + 1)
                plt.imshow(X_reshaped[:, :, index], cmap="gray")
                plt.axis("off")

        plt.show()

    newTest = testing_images.reshape(testing_images.shape[0], -1).T
    pValues = [5, 30]
    for p in pValues:
        accuracy = [0 for i in range(10)]
        curr = np.dot(U[:,:p].T, X_centered)
        neww = []
        coIn = []
        for i in range(10):
            extraFactor = np.eye(p) * 1e-6
            neww.append(np.mean(curr[:,100*i:100*(i+1)], axis = 1, keepdims = True))
            cov = np.cov(curr[:, 100*i:100*(i+1)].T, rowvar=False) + extraFactor
            coIn.append(np.linalg.inv(cov))
        centeredX = newTest - np.mean(newTest, axis = 1, keepdims = True)
        curr_trans = np.dot(U[:,:p].T, centeredX)
        # for currr in zip(Y_trans, testing_images):
        for out,ans in zip(curr_trans.T, testing_labels):
            d = []
            for mu, covv in zip(neww, coIn):
                d.append(QDA(out, mu, covv, 0.1))
            currOutput = np.argmax(d)
            if(currOutput == ans):
                accuracy[currOutput] += 1
        answer = 0
        tests = 0
        for out in accuracy:
            answer += out
        print(f"p : {p}")
        print("accuracies : ")
        for class_label, predictions in enumerate(accuracy):
            class_tests = np.sum(testing_labels == class_label)
            class_acc = predictions / class_tests if class_tests > 0 else 0.0
            tests += class_tests
            print(f"Class {class_label}: {class_acc}")
        acc = answer/tests
        print(f"Overall accuracy : {acc}")
        accuracy = []






# partOne()
getMSE()



