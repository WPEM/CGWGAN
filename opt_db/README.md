
# configure run_all.py

#### Define file paths for the dataset and database
path = file_path = "./your_file_path"
db_path = f"{file_path}/data.db"

#### Instantiate the CIFProcessor and StructureProcessor classes
cif_processor = CIFProcessor(path)
structure_processor = StructureProcessor(file_path, db_path)

#### Process the CIF files to extract and handle the data
cif_processor.process_files()
#### Quickly filter for target systems with reasonable placeholders
#### (The specifics of this filtering would depend on the implementation of `process_files`)

#### Clean up the processed CIF files by removing unreasonable placeholders
#### This helps save disk space and ensures data quality
cif_processor.clean()

#### Process the structures, perform relaxation and phonon calculations, and package the results into the database
structure_processor.process_structures()
#### This step involves executing relaxation and phonon calculations on the structures and storing the results in the database
