#parlamint-taxonomy.py
import pandas
import xml.etree.ElementTree as ET


# Setup Information
namespace = {'tei': 'http://www.tei-c.org/ns/1.0', "xml": "http://www.w3.org/XML/1998/namespace", "XInclude": "http://www.w3.org/2001/XInclude"}

def get_legislature_taxonomy(input_path, export_path):
    """"Given an input_path to the ParlaMint xml legislature taxonomy file, find all instances of house categories and codes and export it as a csv.
    :param input_path: File import path.
    :param export_path: Where to output the path."""
    root = ET.parse(input_path).getroot()
    for elem in root.findall(".//tei:category[@xml:id='parla.organization']", namespace):
        categories = [
            [category.attrib[f"{{{namespace['xml']}}}id"], category.find("tei:catDesc/tei:term", namespace).text] for category in elem.findall("tei:category/tei:category", namespace) + elem.findall("tei:category/tei:category/tei:category", namespace)
        ]
        legislature_taxonomy = pandas.DataFrame(categories, columns = ["parla_tag", "house_type"])
        legislature_taxonomy.to_csv(export_path, index=False)


def get_parlamint_xml_speech(input_path, country_id, dataframe):
    """"

    :param input_path:
    :param country_id:
    :param dataframe:
    """
    root = ET.parse(input_path).getroot()
    for elem in root.findall("tei:text/tei:body/tei:div/tei:u", namespace):
        person_id = f"{country_id}-{elem.attrib['who'][1:]}"
        for seg in elem.findall("tei:seg", namespace):
            speech_id = seg.attrib[f"{{{namespace['xml']}}}id"]
            seg_text = seg.text
            # Not all segments end in a period. To ensure that splitting text by sentence goes well, and that sentences don't combine, I'm adding a period to the end of each segment.
            if seg_text[-1] != ".":
                seg_text += "."
            dataframe.loc[len(dataframe)] = ([speech_id, seg_text, person_id])
    return dataframe

# parlamint_speeches = pandas.DataFrame(columns = ["speech_id", "speech", "person_id"])
# def get_parlamint_xml_speech(input_path, country_id, dataframe):
#     """"
#
#     :param input_path:
#     :param country_id:
#     :param dataframe:
#     """
#     root = ET.parse(input_path).getroot()
#     for elem in root.findall("tei:text/tei:body/tei:div/tei:u", namespace):
#         u_id = elem.attrib[f"{{{namespace['xml']}}}id"]
#         person_id = f"{country_id}-{elem.attrib['who'][1:]}"
#         u_speech = ""
#         for seg in elem.findall("tei:seg", namespace):
#             seg_text = seg.text
#             # Not all segments end in a period. To ensure that splitting text by sentence goes well, and that sentences don't combine, I'm adding a period to the end of each segment.
#             # if seg_text[-1] != [".", "?", "!"]:
#             #     seg_text += "."
#             u_speech += seg_text
#            # # If there is no segment originally added to the total speech, add it with no starting space. If there is, then there will be a starting space (so the sentence is split).
#            #  if not u_speech:
#            #      u_speech += seg_text
#            #  else:
#            #      u_speech += f" {seg_text}"
#         #(?<!\d\.\d)(?<=[.!?])(?!\d)
#         #print(u_speech)
#         #(?!\d)
#         split_u = re.split(r'(?<!hon)(?<=[.!?])(?<![.!?])[^\d.?!]+', u_speech)
#         print(split_u)
#         for i, string in enumerate(split_u):
#             speech = string.strip()
#             speech_id = f"{u_id}-{i+1}"
#             dataframe.loc[len(dataframe)] = ([speech_id, speech, person_id])
#     return dataframe
# parlamint_speeches = get_parlamint_xml_speech(xml_file_paths[0], "GB", parlamint_speeches)
# parlamint_speeches