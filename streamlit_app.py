import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ======= Definir la clase VAE (idéntica a la usada en entrenamiento) =======
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(400, 20)
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        pass  # No se usa en generación

# ======= Cargar modelo entrenado =======
model = VAE()
model.load_state_dict(torch.load("vae_mnist.pth", map_location=torch.device('cpu')))
model.eval()

# ======= Streamlit UI =======
st.title("🧠 Digit Generator")
st.write("This generator uses a VAE model trained from scratch with MNIST.")

# Selector de dígito (aunque no está condicionado, es parte del UI)
digit = st.selectbox("Select the digit (0–9):", list(range(10)))
st.write(f"Showing 5 images generated for the digit {digit} (unconditioned)")

# Generar y mostrar imágenes
fig, axs = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    z = torch.randn(1, 20)  # vector latente aleatorio
    with torch.no_grad():
        gen_img = model.decoder(z).view(28, 28).numpy()
    axs[i].imshow(gen_img, cmap="gray")
    axs[i].axis("off")
st.pyplot(fig)
