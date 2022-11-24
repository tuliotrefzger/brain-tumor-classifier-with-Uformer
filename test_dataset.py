import pathlib
from artefatos import ringing, contrast, blurring, ruido_gaussiano, ghosting
from neural_network import import_nn, define_config, train_valid
from artefatos_testes import teste_artefatos
from process_dataset import (
    generate_dataloader,
    process_dataset,
    process_dataset_train_valid_test,
    generate_dataloader_train_valid_test,
)
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns


def main():
    print(f"TEST INICIALIZED\n\n")

    ## Choosing the device to test.
    if torch.cuda.is_available():
        # device_name = "cuda" # colab
        device_name = "cuda:1"  # servidor
    else:
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Training in: {device_name}\n")

    ## Three classes of tumors (Meningiomas, Gliomas and Pituitary).
    num_classes = 3

    ## Image size either 512x512 or 256x256 works fine.
    img_size = 256

    ## Batch size of 4 images per batch.
    test_batch_size = 4

    ## Choose the artifact you would like to test.
    # artifact = "GaussianNoise"
    # artifact_pretty = "gaussian noise"
    # artifact = "Contrast"
    # artifact_pretty = "contrast"
    # artifact = "Blurring"
    # artifact_pretty = "blurring"
    # artifact = "Ringing"
    # artifact_pretty = "ringing"
    artifact = "Ghosting"
    artifact_pretty = "ghosting"

    ## Importing the model.
    model = import_nn(num_classes, device)
    # Loss function e optimizer.
    criterion, _ = define_config(model, device)

    ## Loading the already trained weights and biases of the model.
    ## You must first run "main.py" to get these weights and biases, which will be stored at
    ## "./resultados/treino_nao_degradadas/" with the name "modelo.pt".
    model_weights_and_biases = "resultados/treino_nao_degradadas/modelo.pt"
    model.load_state_dict(torch.load(model_weights_and_biases))

    ## Accuracy vectors to be plotted.
    acc_degraded = []
    acc_restored = []

    ## Directory where the original dataset is located (change it to the path where your
    ## downloaded version is located if necessary).
    unmodified_dataset_directory = (
        "/mnt/nas/GianlucasLopes/NeuralBlack/patientImages/splits"
    )

    ## Directory containing the degraded images.
    degraded_directory_root = (
        "/mnt/nas/TulioTrefzger/Uformer-for-Artifact-Removal/patientImages/"
    )

    ## The destination directory where the restored images will be stored.
    restored_directory_root = (
        "/mnt/nas/TulioTrefzger/Uformer-for-Artifact-Removal/patientImages/"
    )

    ## Loop to test the accurcaies of the different degradation levels
    for deg_level in range(0, 11):
        print(f"DEGRADATION LEVEL: {deg_level}\n")
        print(
            f"Testing the accuracy of the DEGRADED/UNRESTORED test set of intensity {deg_level}."
        )
        ## Path to the original test dataset (without any artifacts).
        if deg_level == 0:
            degraded_directory = unmodified_dataset_directory + "/test"
        ## Path to each degraded test dataset.
        elif deg_level < 10:
            degraded_directory = (
                degraded_directory_root
                + artifact
                + "/splits"
                + artifact
                + "0"
                + str(deg_level)
                + "/test"
            )
        else:
            degraded_directory = (
                degraded_directory_root
                + artifact
                + "/splits"
                + artifact
                + str(deg_level)
                + "/test"
            )

        test_set = process_dataset(
            dataset_path=degraded_directory,
            img_size=img_size,
            funcao_geradora_artefato=None,
            nivel_degradacao=0,
            nivel_aleatorio_teto=0,
            nivel_aleatorio=0,
            num_classes=num_classes,
            augmentation=False,
        )

        test_gen = generate_dataloader(
            dataset=test_set, batch_size=test_batch_size, balancear_dataset=False
        )

        ## Directory where the resulting information and graphs will be stored.
        path_save_unrestored_results = (
            "resultados/"
            + "TulioTrefzger/"  # Just the directory name that came to mind. Perhaps change it to yours.
            + artifact
            + "/WithoutUformer/"
        )
        if deg_level < 10:
            path_save_unrestored_results += "0"
        path_save_unrestored_results += str(deg_level) + "/"

        ## Function that plots the confusion matrices and gets the accuracy.
        acc, _ = test_model(
            model, test_gen, criterion, device, path_save_unrestored_results
        )
        acc_degraded.append(acc)
        print(
            f"Ending testing with the DEGRADED/UNRESTORED test set of intensity {deg_level}.\n"
        )

        print(
            f"Testing the accuracy of the RESTORED test set of intensity {deg_level}."
        )
        if deg_level < 10:
            restored_directory = (
                restored_directory_root
                + artifact
                + "/splitsAdjustedUformer0"
                + str(deg_level)
                + "/test"
            )
        else:
            restored_directory = (
                restored_directory_root
                + artifact
                + "/splitsAdjustedUformer"
                + str(deg_level)
                + "/test"
            )

        test_set = process_dataset(
            dataset_path=restored_directory,
            img_size=img_size,
            funcao_geradora_artefato=None,
            nivel_degradacao=0,
            nivel_aleatorio_teto=0,
            nivel_aleatorio=0,
            num_classes=num_classes,
            augmentation=False,
        )

        test_gen = generate_dataloader(
            dataset=test_set, batch_size=test_batch_size, balancear_dataset=False
        )

        path_save_restored_results = (
            "resultados/"
            + "TulioTrefzger/"  # Just the directory name that came to mind. Perhaps change it to yours.
            + artifact
            + "/AdjustedWithUformer/"
        )
        if deg_level < 10:
            path_save_restored_results += "0"
        path_save_restored_results += str(deg_level) + "/"

        ## Function that plots the confusion matrices and gets the accuracy.
        acc, _ = test_model(
            model, test_gen, criterion, device, path_save_restored_results
        )
        acc_restored.append(acc)
        print(f"Ending testing with the RESTORED test set of intensity {deg_level}.\n")
        print()
        plt.close("all")

    ## Plotting the resulting accuracy vectors into a single image
    deg_level = np.linspace(0, 10, 11)
    plt.figure()
    plt.plot(deg_level, acc_degraded, color="orange", label="Without Uformer")
    plt.plot(deg_level, acc_restored, "b", label="With Uformer")
    plt.title("Artifact: " + artifact_pretty)
    plt.xlabel("Degradation level")
    plt.ylabel("Accuracy (%)")
    plt.xticks(deg_level)
    plt.legend()
    plt.grid()
    path_save_accuracy_plt = (
        "resultados/" + "TulioTrefzger/" + artifact + "/deg_level_vs_acc.png"
    )
    plt.savefig(path_save_accuracy_plt, bbox_inches="tight")

    print(f"\nTEST ENDED")

    plt.close("all")


def test_model(model, test_gen, criterion, device, path_salvar_modelo, show_info=True):

    # Local onde o modelo treinado foi salvo (assume-se que o nome é modelo.pt)
    pathlib_salvar_modelo = pathlib.Path(path_salvar_modelo)
    # Criando pasta caso não exista
    pathlib_salvar_modelo.mkdir(parents=True, exist_ok=True)

    if show_info:
        # logger.info("\n\nIniciando teste\n\n")
        print("\nTest inicialized\n")

    # Modo de teste
    model.eval()

    loss = 0

    # contador de imagens totais
    total_images = 0

    # Impede o cálculo de gradientes, poupa memória e tempo
    with torch.no_grad():
        correct = 0
        labels = []
        pred = []

        # Teste
        for X, y in test_gen:
            # mandando imagens e labels para a gpu
            X, y = X.to(device), y.to(device)

            # Guardando os labels originais para visualizar a matriz de confusão
            labels.append(torch.argmax(y, dim=1).data)

            # Predict
            y_val = model(X)

            # get argmax of predicted values, which is our label
            predicted = torch.argmax(y_val, dim=1).data
            pred.append(predicted)

            loss += criterion(
                y_val.float(), torch.argmax(y, dim=1).long()
            ).item() * X.size(0)

            # número de acertos
            correct += (predicted == torch.argmax(y, dim=1)).sum()

            # Contando imagens
            total_images += y.shape[0]
    loss /= len(test_gen.dataset)

    if show_info:

        # logger.info(f"Test Loss: {loss:.4f}")
        # logger.info(f'Test accuracy: {correct.item()*100/(total_images):.2f}%')
        print(f"Test Loss: {loss:.4f}")
        print(f"Test accuracy: {correct.item()*100/(total_images):.2f}%")

        # Convert list of tensors to tensors -> Para usar nas estatísticas
        labels = torch.stack(labels)
        pred = torch.stack(pred)

        # Define ground-truth labels as a list
        LABELS = ["Meningioma", "Glioma", "Pituitary"]

        # Plot the confusion matrix
        arr = confusion_matrix(labels.view(-1).cpu(), pred.view(-1).cpu())
        df_cm = pd.DataFrame(arr, LABELS, LABELS)
        plt.figure(figsize=(9, 6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="viridis")
        plt.xlabel("Prediction")
        plt.ylabel("Target")
        plt.savefig(f"{path_salvar_modelo}confusion matrix.png", bbox_inches="tight")

    return correct.item() * 100 / (total_images), loss


if __name__ == "__main__":
    # Para conseguir reproduzir resultados
    torch.manual_seed(42)
    np.random.seed(42)
    plt.ioff()  # Desabilita o modo interativo do matplotlib
    main()
