
### Configuration for Running `gen_crystal.py`

To run `gen_crystal.py`, you need to configure the `gen.ini` file. Below is a detailed breakdown of the required settings:

**Data Generator Settings:**

- **data_path:** The path to the folder where the data is stored.
- **generator_model_path:** The path to the trained model file. Uncomment the second option if you wish to use a different model.
- **z_dim:** The dimension of the latent space vector.
- **output_dim:** The dimension of the output vector.

**General Settings:**

- **num_samples_tryto_generate:** The number of samples to attempt to generate.

**Structure Generator Settings:**

- **spgn:** The space group number.
- **elemset:** A comma-separated list of elements used in generation.
- **N_max_per_cell:** The maximum number of atoms per unit cell.

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
```

If you wish to use our pretrained model, please contact us.
