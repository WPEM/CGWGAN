
# CGWGAN | [Paper](https://www.oaepublish.com/articles/jmi.2024.24?utm_campaign=website&utm_medium=email&utm_source=sendgrid.com)

## Crystal Generative Framework based on Wyckoff Generative Adversarial Network

We present the Crystal Generative Framework based on the Wyckoff Generative Adversarial Network (CGWGAN). CGWGAN utilizes a strategy that focuses on generating crystal templates while effectively masking the occupancy information of elements at specific sites within the crystal structure.

## Resources

- **Crystal templates**: Available on [Hugging Face](https://huggingface.co/datasets/caobin/CGWGAN).
- **Novel crystal data**: Available on [Figshare](https://doi.org/10.6084/m9.figshare.26888884.v1).
- **CGWGAN generator**: Located in the 'model' folder.
- **Atom infill and high-throughput filter**: Found in the 'opt_db' folder.

### Prerequisites

- Ensure that the following packages are installed: `phonopy`, `pymatgen`, `ase`, and a surrogate model such as `m3gnet`.

### Example Setup

- This example uses `m3gnet` as the surrogate model.
- Provide the path to the database that stores structures with substituted elements.
- Specify this in the `./opt_db/run_all.py` file:

```python
file_path = "path_2_db"
db_path = f"{file_path}/data.db"
cif_processor = CIFProcessor(file_path)
structure_processor = StructureProcessor(file_path, db_path)
cif_processor.process_files()
cif_processor.clean()
structure_processor.process_structures()
```

## Contact Information

- **Mr. SU Tianhao**  
  Email: thsu0407@gmail.com

- **Mr. Cao Bin**  
  Email: bcao686@connect.hkust-gz.edu.cn

## Acknowledgement

If you utilize the data or code from this repository, please reference [our paper](https://www.oaepublish.com/articles/jmi.2024.24?utm_campaign=website&utm_medium=email&utm_source=sendgrid.com).

```
@article{su2024cgwgan,
  title={CGWGAN: crystal generative framework based on Wyckoff generative adversarial network},
  author={Su, Tianhao and Cao, Bin and Hu, Shunbo and Li, Musen and Zhang, Tong-Yi},
  journal={Journal of Materials Informatics},
  volume={4},
  number={4},
  pages={N--A},
  year={2024},
  publisher={OAE Publishing Inc.}
}
```
