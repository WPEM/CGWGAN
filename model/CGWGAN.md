### Configuration for Running gen_crystal.py

- To run `gen_crystal.py`, you need to configure the `gen.ini` file. Here is a detailed breakdown of the required settings:

```ini
[data_generator]
data_path = ../database/
generator_model_path = ../database/generator.pth
# generator_model_path = generator_addnumber_2sa.pth
z_dim = 128
output_dim = 20

[general]
num_samples_tryto_generate = 1000

[stru_generator]
spgn = 9
elemset = H,H,H,H
N_max_per_cell = 200

Data Generator Settings:

data_path: Path to the folder where the data is stored.
generator_model_path: Path to the trained model file. You can uncomment the second option to use a different model.
z_dim: Dimension of the latent space vector.
output_dim: Dimension of the output vector.
General Settings:

num_samples_tryto_generate: Number of samples to attempt to generate.
Structure Generator Settings:

spgn: Space group number.
elemset: Comma-separated list of elements used in generation.
N_max_per_cell: Maximum number of atoms per unit cell.
