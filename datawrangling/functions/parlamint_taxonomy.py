#parlamint_taxonomy.py
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
        legislature_taxonomy.to_csv(f"{export_path}\\legislature_taxonomy.csv", index=False)

