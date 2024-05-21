#parlamint_speech_sorting.py
import pandas
import xml.etree.ElementTree as ET

# Setup Information
namespace = {'tei': 'http://www.tei-c.org/ns/1.0', "xml": "http://www.w3.org/XML/1998/namespace", "XInclude": "http://www.w3.org/2001/XInclude"}

def get_parlamint_xml_speech(input_path, country_id = ""):
    """"
    Given an input path to an xml ParlaMint file, export a pandas dataframe with its speeches, speech id, and a person id, potentially modified per country.
    :param input_path: Input file path.
    :param country_id: Country_id (i.e. "GB") to add at the end of the person_id in the dataframe.
    :return: A pandas dataframe with speeches, speech id, and person id.
    """
    # Setup - Getting root and dict info
    root = ET.parse(input_path).getroot()
    export_speech_dict = {"speech_id": [], "speech": [], "person_id": []}

    # Searching through all tei speeches
    for elem in root.findall("tei:text/tei:body/tei:div/tei:u", namespace):
        export_speech_dict["speech_id"].append \
            (elem.attrib[f"{{{namespace['xml']}}}id"]) # Add speech_id
        try: # Add person_id (checking if country_id exists, and if an error (as a person_id may not exist) add a blank category.
            if country_id:
                export_speech_dict["person_id"].append(f"{country_id}-{elem.attrib['who'][1:]}")
            else:
                export_speech_dict["person_id"].append(elem.attrib['who'][1:])
        except KeyError:
            export_speech_dict["person_id"].append("")

        # Sorting through speech segments.
        u_speech = ""
        for seg in elem.findall("tei:seg", namespace):
            seg_text = seg.text
            # Not all segments end in a period. To ensure that splitting text by sentence goes well, and that sentences don't combine, a period is added to the end of each segment.
            if seg_text[-1] not in [".", "?", "!"]:
                seg_text += "."
            # If there is no segment originally added to the total speech, add it with no starting space. If there is, then there will be a starting space (so the sentence is split).
            if not u_speech:
                u_speech += seg_text
            else:
                u_speech += f" {seg_text}"
        export_speech_dict["speech"].append(u_speech) # Adding speech
    return pd.DataFrame(export_speech_dict)

def get_corpus_list_speeches(corpus_list, input_path, export_path):
    """
    Given ParlaMint corpuses, output a csv and parquet file containing all of their speeches..
    :param corpus_list: The corpus that are used to get ParlaMint data (i.e. "ParlaMint-GB", since all files begin with this).
    :param input_path: Where the files come from.
    :param export_path: Where the files should be outputted.
    """
    # Begin by looking through the corpus list.
    for corpus in corpus_list:
        # Base info, adjusted later
        xml_file_paths = []
        parlamint_speeches = pandas.DataFrame(columns = ["speech_id", "speech", "person_id"])

        # Get all of the xml file paths from the xml file.
        base_xml_path = f"{input_path}\\{corpus}\\{corpus}.TEI"
        xml_info_path = Path(f"{base_xml_path}\\{corpus}.xml")
        root = ET.parse(xml_info_path).getroot()
        for elem in root.findall("XInclude:include", namespace): # Loop through the root to find them
            xml_file_paths.append(Path(f"{base_xml_path}\\{elem.attrib["href"]}"))

        # With all of the xml file paths, begin to search through all links to find all speeches, before concating the together.
        number_xmls = len(xml_file_paths)
        print(f"Beginning {corpus} search with {number_xmls} files.")
        for i, file_path in enumerate(xml_file_paths):
            # print(f"Finished search {i+1} out of {number_xmls}.") # Lets you see how many instances are being fulfilled.
            new_speech = get_parlamint_xml_speech(file_path, corpus[-2:])
            parlamint_speeches = pd.concat([parlamint_speeches, new_speech])

        # Export to csv and parquet.
        parlamint_speeches.to_csv(f"{export_path}\\{corpus}-RawSpeechesCSV.csv", index=False)
        parlamint_speeches.to_parquet(f"{export_path}\\{corpus}-RawSpeechesParquet.gzip", index=False, compression='gzip')

