
# CGWGAN

## Crystal Generative Framework based on Wyckoff Generative Adversarial Network

In this study, we present the Crystal Generative framework based on the Wyckoff Generative Adversarial Network (CGWGAN). CGWGAN employs a strategy focused on the generation of crystal templates, effectively masking the occupancy information of elements at specific sites within the crystal.

This repository provides the essential code for atom infill and phonon spectrum calculations, which are crucial for supporting CGWGAN.

All templates with 3-4 asymmetric units generated in our work are available as open-source resources on the [datasets CGWGAN](https://huggingface.co/datasets/caobin/CGWGAN).

## Contact Information:

Mr. SU Tianhao  

Email: thsu0407@gmail.com

Mr. Cao Bin  

Email: bcao686@connect.hkust-gz.edu.cn



## Acknowledgement:
If you utilize the data or code from this repository, please reference our paper (currently unpublished).


### Prerequisites

- Install `phonopy`, `pymatgen`, `ase`, and the surrogate model such as `m3gnet`.

### Example Setup

- This example uses `m3gnet` as the surrogate model.
- Provide the path to the database storing structures with substituted elements.
- Specify this in the `run_all.py` file:

```python
file_path = "path_2_db"
db_path = f"{file_path}/data.db"
cif_processor = CIFProcessor(file_path)
structure_processor = StructureProcessor(file_path, db_path)
cif_processor.process_files()
cif_processor.clean()
structure_processor.process_structures()

