
The `run_all.py` script is designed to streamline the processing of CIF (Crystallographic Information File) files and the subsequent handling of structural data. This script performs several key tasks, including processing CIF files, filtering and cleaning data, and conducting structural analyses. The results are then stored in a SQLite database for further use.

## File Paths Configuration

```python
#### Define file paths for the dataset and database
path = file_path = "./your_file_path"
db_path = f"{file_path}/data.db"
```

- **`path`**: Specifies the location of the CIF files. Replace `"./your_file_path"` with the actual path to your CIF files.
- **`db_path`**: Defines the path for the SQLite database where the results will be stored. It is constructed by appending `"data.db"` to the `file_path`.

## Class Instantiation

```python
#### Instantiate the CIFProcessor and StructureProcessor classes
cif_processor = CIFProcessor(path)
structure_processor = StructureProcessor(file_path, db_path)
```

- **`CIFProcessor`**: This class is responsible for handling and processing CIF files. It will read the CIF files from the specified `path`.
- **`StructureProcessor`**: This class processes the structural data extracted from CIF files. It performs calculations and stores results in the database specified by `db_path`.

## Processing CIF Files

```python
#### Process the CIF files to extract and handle the data
cif_processor.process_files()
```

- **`process_files()`**: This method processes the CIF files located at `path`. The specifics of this processing depend on the implementation of the `CIFProcessor` class. Generally, it involves extracting data from CIF files and preparing it for further analysis.


## Cleaning Processed CIF Files

```python
#### Clean up the processed CIF files by removing unreasonable placeholders
#### This helps save disk space and ensures data quality
cif_processor.clean()
```

- **`clean()`**: This method removes unnecessary or unreasonable placeholders from the CIF files. Cleaning helps in saving disk space and maintaining the quality of the data by eliminating irrelevant or corrupted entries.

## Processing Structures and Calculations

```python
#### Process the structures, perform relaxation and phonon calculations, and package the results into the database
structure_processor.process_structures()
```

- **`process_structures()`**: This method handles the processing of structural data. It involves:
  - Performing relaxation calculations to optimize the structure.
  - Conducting phonon calculations to study the vibrational properties of the structures.
  - Storing the results of these calculations in the SQLite database specified by `db_path`.



