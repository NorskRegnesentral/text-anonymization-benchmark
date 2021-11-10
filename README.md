The _Text Anonymization Benchmark_ (TAB) is a new, open-source corpus for text anonymization. It comprises 1,268 English-language court cases from the [European Court of Human Rights (ECHR)](https://www.echr.coe.int/Pages/home.aspx?p=home) manually annotated with:
* semantic categories for personal identifiers,
* masking decisions (in regard to the re-identification risk for the person to protect),
* confidential attributes,
* co-reference relations.


## General information

This repository contains the v1.0 release of the Text Anonymization Benchmark, a corpus for text anonymization.
The corpus comprises 1,268 English-language court cases from the [European Court for Human Rights (ECHR)](https://www.echr.coe.int/). The documents were manually annotated with information about personal identifiers (including their semantic category and need for masking), confidential attributes and co-reference relations. 

## Data format
The data is distributed in a standoff JSON format consisting of a list of document object with the following information:

| Variable name | Description |
|---------------|-------------|
| annotations | an object with document annotations, each containing an object with entity mention annotations |
| dataset_type | which data split the court case belongs to (train /dev / test) |
| doc_id | the ID of the court case (e.g. “001-61807”) |
| meta | an object with metadata for each case (year, countries and legal articles involved etc.) |
| quality_checked | whether the document was revised by another annotator |
| task | the target of the anonymisation task (i.g. who to anonymise) |
| text | the text of the court case used during the annotation |

Each entity mention object under 'annotations' has the following attributes:

| Variable name | Description |
|---------------|-------------|
| entity_type | the semantic category of the entity (e.g. PERSON) |
| entity_mention_id | ID of the entity mention |
| start_offset | start character offset of the annotated span |
| end_offset | end character offset of the annotated span |
| span_text | the text of the annotated span |
| edit_type | type of annotator action for the mention (check / insert / correct) |
| identifier_type | the need for masking, masked if 'DIRECT' or 'QUASI', 'NO_MASK' otherwise |
| entity_id | ID of the entity the entity mention is related to in meaning |
| confidential_status | category of a potential source of discrimination (e.g. beliefs, sexual orientation etc.) |

## License

skweak is released under an MIT License.

The MIT License is a short and simple permissive license allowing both commercial and non-commercial use of the software. The only requirement is to preserve the copyright and license notices (see file [License](https://github.com/NorskRegnesentral/text-anonymisation-benchmark/blob/master/LICENSE.txt)). Licensed works, modifications, and larger works may be distributed under different terms and without source code.

 
