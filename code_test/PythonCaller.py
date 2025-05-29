import numpy as np
import pandas as pd
import re


def processFeature(feature):
    pass


# class to rename the header names
class Rename_functions:
    # Dictionary to rename file headers
    def __init__(self):
        self.rename_map_key = {
            "MDRKB": "MD",
            "MDDF": "MD",
            "MDML": "MD",
            "MDRT": "MD",
            "TIME": "OWT",
            "MEASUREDDEPTH": "MD",
            "TWOWAYTIME": "TWT",
            "TWOWAYVERTICALTIME": "TWT",
            "TVDD": "TVD",
            "TVDMSL": "TVDSS",
            "TVDSSUBSEA": "TVDSS",
            "TWTT": "TWT",
            "TWOWAY": "TWT",
            "TWOWAYTRAVELTIME": "TWT",
            "TVDSD": "TVDSS",
            "TWTMSL": "TWT",
            "OWTRD": "OWT",
            "OWTREFML": "OWT",
            "OWTMSL": "OWT",
            "TC": "OWT",
            "One-Way Vertical Time": "OWT",
            "One Way Time": "OWT",
            "Two Way Time": "TWT",
        }
        # Dictionary to rename depth reference datum
        self.rename_depth_reference_datum = {
            "TVDMSL": "MSL",
            "TVDSUBSEA": "MSL",
            "TVDSD": "MSL",
        }
        # Dictionary to rename depth reference point
        self.rename_depth_reference_point = {
            "MD-RKB": "RKB",
            "MDRKB": "RKB",
            "MDDF": "DF",
            "MDRT": "RT",
        }

    def rename_key(self, key):
        key = (
            key.upper()
            .replace("(M)", "")
            .replace("(MS)", "")
            .replace("(FT)", "")
            .replace("(S)", "")
        )
        chars = "`*_>-#+.!$() "
        for c in chars:
            key = key.replace(c, "")
        return self.rename_map_key.get(key, key)

    def depth_reference_datum(self, key):
        key = (
            key.upper()
            .replace("(M)", "")
            .replace("(MS)", "")
            .replace("(FT)", "")
            .replace("(S)", "")
        )
        chars = "`*_>#+-.!$() "
        for c in chars:
            key = key.replace(c, "")
        return self.rename_depth_reference_datum.get(key, None)

    def depth_reference_point(self, key):
        key = (
            key.upper()
            .replace("(M)", "")
            .replace("(MS)", "")
            .replace("(FT)", "")
            .replace("(S)", "")
        )
        chars = "`*_>#+-.!$() "
        for c in chars:
            key = key.replace(c, "")
        return self.rename_depth_reference_point.get(key, None)

    def interpret_time(self, key):
        key = key


class FeatureProcessor:
    """Template Class Interface:
    When using this class, make sure its name is set as the value of the 'Class
    to Process Features' transformer parameter.
    """

    def __init__(self):
        """Base constructor for class members."""

    # function to try different file encodings when reading the file.
    def encodings(self, filepath):
        encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    filepath, header=None, encoding=encoding, on_bad_lines="skip"
                )
                break
            except UnicodeDecodeError:
                print(
                    f"Error reading {filepath} using utf-8 encoding. Trying another encoding..."
                )
                pass
        return df

    def extract_data(self, filepath, text_or_number):
        if filepath.lower().endswith(".txt"):
            try:
                with open(filepath, "r") as file:
                    contents = file.read()
            except FileNotFoundError:
                print("File not found.")
            except IOError:
                print("An I/O error occurred.")
            my_list = contents.split("\n")[1:]
            # it was seen that many .txt files contained Measured Depth (MSL) that were not right. It doesn't apply for Norway.
            string_list = [
                test.replace("\\t", " ")
                .replace("NaN", "")
                .replace("Measured Depth (MSL)", "None")
                .replace("Measured Depth", "MD")
                .replace(",", "  ")
                for test in my_list
            ]
        elif filepath.lower().endswith(".asc"):
            try:
                df = pd.read_csv(filepath, header=None)
                if pd.errors.ParserError:
                    df = pd.read_csv(
                        filepath, header=None, on_bad_lines="skip"
                    ).drop_duplicates(ignore_index=True)
            except Exception:
                # Sometimes there are different encodings and the file cannot be read with "utf-8" encoding.
                df = self.encodings(filepath=filepath).drop_duplicates(
                    ignore_index=True
                )
                if df is None:
                    print(filepath + " - File cannot be red")
            my_list = df.to_string(index=False).split("\n")[1:]
            string_list = [
                test.replace("\\t", " ").replace("NaN", "") for test in my_list
            ]

        elif filepath.lower().endswith(".xlsx"):
            df = pd.read_excel(filepath, header=None)
            my_list = [",".join(str(x) for x in row) for row in df.values]
            if text_or_number == "text":
                string_list = [
                    test.replace(",", "  ").replace("nan", ".") for test in my_list
                ]
            elif text_or_number == "number":
                string_list = [
                    test.replace(",", "  ").replace("nan", "-999.25")
                    for test in my_list
                ]

        return string_list

    # Function to identify text features in the text

    def read_text(self, filepath):
        string_list = self.extract_data(filepath=filepath, text_or_number="text")

        # It uses regular expression (regex) to check if there are at least two headers. There can be units within these headers
        regex_pattern = r"(?P<GP1>TVD SS|TVDSS|TVD MSL|OWTSRD|TVDMSL|TVDSRD|MD RKB|MDRKB|TVDSD|RKB|OWT|TWT|TVD|MD|MSL|Depth|One Way Time|Two Way Time)\)?(?P<GP1_unit>\s?\(?(m|s|ms)?\)?)?\s+\(?(?P<GP2>TVD SS|TVDSS|TVD MSL|OWTSRD|TVDMSL|TVDSRD|MD RKB|MDRKB|TVDSD|RKB|OWT|TWT|TVD|MD|MSL|Depth|One Way Time|Two Way Time)\)?(?P<GP2_unit>\s?\(?(m|s|ms)\)?)?\s?\(?(?P<GP3>TVD SS|TVDSS|TVD MSL|OWTSRD|TVDMSL|TVDSRD|MD RKB|MDRKB|TVDSD|RKB|OWT|TWT|TVD|MD|MSL|Depth|One Way Time|Two Way Time)?"
        match_string = ""
        match_string_units = ""
        for string in string_list:
            match = re.search(regex_pattern, string, re.IGNORECASE)

            if match:
                match_string = string.replace(")", ") ")

                match_string = match_string.strip()
                pattern = r"\s{2,}"
                match_string = re.split(pattern, match_string)
                match_string = [match.upper() for match in match_string]
                if len(match_string[0]) == 1:
                    try:
                        pattern = r"\s{1,}"
                        match_string = re.split(pattern, match_string[0])
                        match_string = [match.upper() for match in match_string]
                    except Exception:
                        print(f"{filepath} - Headers cannot be identified")
                        pass
            else:
                pass

        # Regex to idenfity if there are lines with units alone
        for string in string_list:
            regex_pattern_units = r"(?<=\s|\]|\)|\[|\()(?P<GP1>[m|s|ms|msec|s|m/s|ft])\s*\]?\)?\s+\(?\[?\s*(?P<GP2>[m|s|ms|msec|s|m/s|ft])\s*\]?\)?\s*\(?\[?\s*(?P<GP3>m|s|ms|msec|s|m/s|ft])?\s*(?=\s|\]|\)|\[|\()"
            match_units = re.search(regex_pattern_units, string, re.IGNORECASE)
            if match_units:
                match_string_units = string.replace(")", ") ").strip()
                match_string_units = match_string_units.strip()

                pattern_units = r"\s{2,}"
                match_string_units = re.split(pattern_units, match_string_units)

                if len(match_string_units) == 1:
                    try:
                        pattern_units = r"\s{1,}"
                        match_string_units = re.split(
                            pattern_units, match_string_units[0]
                        )
                    except Exception:
                        # Lines of units were not identified alone.
                        pass
                else:
                    pass
            else:
                pass

        # It tries to identify if there are units within parenthesis or brackets together with the header
        if len(match_string_units) == 0:
            match_string_units = []
            try:
                pattern = r"\(.*?\)"
                for item in match_string:
                    match_units_parenthesis = re.findall(pattern, item, re.IGNORECASE)
                    if match_units_parenthesis:
                        match_units_parenthesis = match_units_parenthesis[0].lower()
                        match_string_units.append(match_units_parenthesis)

                    else:
                        match_string_units.append("")
                        pass
            except Exception:
                # No unit was found within parenthesis
                pass

        if all(item == "" for item in match_string_units):
            try:
                pattern = r"\[(.*?)\]"
                for item in match_string:
                    match_units_brackets = re.findall(pattern, item, re.IGNORECASE)

                    if match_units_brackets:
                        match_units_brackets = match_units_brackets[0].lower()
                        match_string_units.append(match_units_brackets)
                    else:
                        pass
            except Exception:
                # No unit was found within brackets
                pass
        else:
            pass

        # Regex to find acquisition depth elevation
        for string in string_list:
            regex_depth_elevation = r"(?:elev|elevation|DF|DEPTH)\s+(of)?\s?(?:depth|MD|REFERENCE|elevation)\s?(datu)?"
            match_depth_elevation = re.search(
                regex_depth_elevation, string, re.IGNORECASE
            )

            if match_depth_elevation:
                regex_depth_elevation = r"(?:-?\d+(?:\.\d*)?)"
                match_depth_elevation_number = re.findall(regex_depth_elevation, string)
                regex_depth_elevation_unit = r"(?:\W|\d|\s)(m|ft)"
                match_depth_elevation_unit = re.findall(
                    regex_depth_elevation_unit, string
                )

                if (
                    len(match_depth_elevation_number) == 2
                    and len(match_depth_elevation_unit) == 2
                ):
                    try:
                        match_depth_elevation_number = match_depth_elevation_number[0]
                        match_depth_elevation_unit = match_depth_elevation_unit[0]
                    except Exception:
                        # No depth elevation number was found
                        pass
                else:
                    pass
            else:
                pass
        # Regex to find acquisition depth datum
        for string in string_list:
            regex_depth_reference_datum = r"^(?!.*elev).*\b\s?depth\s+datu"
            match_depth_reference_datum = re.search(
                regex_depth_reference_datum, string, re.IGNORECASE
            )
            if match_depth_reference_datum:
                regex_msl = r"\bmsl\b"
                match_msl = re.search(regex_msl, string, re.IGNORECASE)
                if match_msl:
                    depth_reference_datum_text = "MSL"
                    break
                else:
                    pass
                depth_reference_datum_text = string
                chars = "`*_>#+-.!$(): "
                for c in chars:
                    depth_reference_datum_text = (
                        depth_reference_datum_text.upper()
                        .replace("DEPTH", "")
                        .replace("DATUM", "")
                        .replace(c, "")
                    )
                rename_depth_reference_datum = {
                    "TVDMSL": "MSL",
                    "TVDSD": "MSL",
                    "MDRKB": "RKB",
                }
                depth_reference_datum_text = rename_depth_reference_datum.get(
                    depth_reference_datum_text, depth_reference_datum_text
                )
                break
            else:
                pass

        if "match_depth_elevation_number" not in locals():
            match_depth_elevation_number = None
        else:
            match_depth_elevation_number = (
                str(match_depth_elevation_number)
                .replace("[", "")
                .replace("'", "")
                .replace("]", "")
            )

        try:
            if match_depth_elevation_number is not None:
                number = float(match_depth_elevation_number)
                # if depth of reference is zero it means that the reference is not the kelly bushing. Then we set it to None and take the value from SMDA later in the
                # workflow. If it is higher than 100 it means that it is not kelly bushing either, and we set it to None.
                # This is why is risky to retrieve data from onshore wells, because the reference of elevation changes and it is generally not well documented.
                if number == 0 or number > 100:
                    match_depth_elevation_number = None
        except ValueError:
            # Handle invalid string input
            if match_depth_elevation_number == "0":
                match_depth_elevation_number = None

        # if there was not a match for depth elevation unit or depth of reference datum, these will be set to None
        if "match_depth_elevation_unit" not in locals():
            match_depth_elevation_unit = None
        else:
            match_depth_elevation_unit = (
                str(match_depth_elevation_unit)
                .replace("[", "")
                .replace("'", "")
                .replace("]", "")
            )

        if "depth_reference_datum_text" not in locals():
            depth_reference_datum_text = None
        else:
            depth_reference_datum_text = (
                str(depth_reference_datum_text)
                .replace("[", "")
                .replace("'", "")
                .replace("]", "")
            )

        # it puts the header (TVD, TWT...) and the units in a dictionary
        dict_match = dict()
        for i in range(0, len(match_string_units)):
            match_string_units[i] = match_string_units[i].lower()
            match_parenthesis = re.findall(r"\(([^)]+)\)", match_string_units[i])
            # If units are within parenthesis or brackets, it will take them out
            if match_parenthesis:
                match_string_units[i] = match_parenthesis[0]
            else:
                pass
            match_key = re.findall(r"\[([^\]]+)\]", match_string_units[i])
            if match_key:
                match_string_units[i] = match_key[0].lower()
            else:
                pass

        # It puts all text data in one dictionary
        dict_match["match_header"] = match_string
        dict_match["match_units"] = match_string_units
        dict_match["depth_elevation_number"] = match_depth_elevation_number
        dict_match["depth_elevation_unit"] = match_depth_elevation_unit
        dict_match["depth_reference_datum_text"] = depth_reference_datum_text
        return dict_match

    def confirm_units(self, dict_match):
        # It tries to match the header and the units
        if len(dict_match["match_header"]) == len(dict_match["match_units"]):
            units = {"measure": "unit"}
            for i in range(len(dict_match["match_header"])):
                units[dict_match["match_header"][i]] = dict_match["match_units"][i]
        return units

    def read_data(self, filepath):
        # Tries to extract numeric columns
        string_list = self.extract_data(filepath=filepath, text_or_number="number")

        regex_pattern_number = r"[0-9.]+\s+[0-9.]+\s*[0-9.]?"
        data = []
        for string in string_list:
            match = re.search(regex_pattern_number, string, re.IGNORECASE)
            if match:
                match_string = string.strip()
                pattern = r"\s{1,}"
                match_string = re.split(pattern, match_string)

                # match_string = [try float(x) else x for x in match_string]
                match_string = [float(x) if x.isnumeric() else x for x in match_string]
                data.append(match_string)
            else:
                pass

        return data

    def verification(self, data, regex):
        # verifies that all numeric columns can be associated to a header. Correct if this association is True.
        try:  # Check if all sublists have the same length
            x = np.array(data, dtype=object)
        except Exception:
            print("no")
            pass
        dictionary = dict()
        embed()
        if len(regex) == x.shape[1]:
            correct = "Correct"
        else:
            correct = "Incorrect"
        if correct == "Correct":
            for i in range(0, len(regex)):
                dictionary[regex[i]] = x[:, i].tolist()

        return dictionary

    def specific_assignment(self, data, regex):
        pass

    # def input(self, feature):  # Main starting point
    def input(self, filepath):  # Main starting point
        # filepath = feature.getAttribute("file_path")
        filepath = filepath
        # filepath = filepath
        try:
            qc_description = list()
            # Import headers and numeric columns

            dict_match = self.read_text(filepath=filepath)

            regex_units = dict_match["match_units"]
            regex_units = str(regex_units)
            # feature.setAttribute("regex_units", regex_units)
            regex_depth_elevation_number = dict_match["depth_elevation_number"]
            # feature.setAttribute(
            #    "depth_reference_elevation", regex_depth_elevation_number
            # )
            regex_depth_elevation_unit = dict_match["depth_elevation_unit"]
            # feature.setAttribute(
            #    "depth_reference_elevation_unit", regex_depth_elevation_unit
            # )

            regex_depth_reference_datum_text = dict_match["depth_reference_datum_text"]
            data = self.read_data(filepath=filepath)

            dict_verified = self.verification(
                regex=dict_match["match_header"], data=data
            )

            renamer = Rename_functions()
            # It will rename all the headers
            for key, value in dict_verified.items():
                depth_reference_datum_header = renamer.depth_reference_datum(key)
                if depth_reference_datum_header is not None:
                    break
            # The code extracts depth reference datum both from the text and header. It will give preference to the depth reference in header.
            if (
                depth_reference_datum_header is not None
                and regex_depth_reference_datum_text is not None
            ):
                depth_reference_datum = depth_reference_datum_header

            elif (
                depth_reference_datum_header is not None
                and regex_depth_reference_datum_text is None
            ):
                depth_reference_datum = depth_reference_datum_header

            elif (
                depth_reference_datum_header is None
                and regex_depth_reference_datum_text is not None
            ):
                depth_reference_datum = regex_depth_reference_datum_text

            else:
                depth_reference_datum = None

            # feature.setAttribute("depth_reference_datum", depth_reference_datum)
            # It will rename all depth_reference points
            for key, value in dict_verified.items():
                depth_reference_point = renamer.depth_reference_point(key)
                if depth_reference_point is not None:
                    break

            if "depth_reference_point" not in locals():
                depth_reference_point = None
            # feature.setAttribute("depth_reference_point", depth_reference_point)
            # It will rename all the headers
            for key, value in dict_verified.items():
                key_renamed = renamer.rename_key(key)
                dict_verified = {
                    key_renamed if k == key else k: v for k, v in dict_verified.items()
                }
                if key != key_renamed:
                    qc_description.append("{} renamed to {}.".format(key, key_renamed))
                if key == "TIME":
                    qc_description.append("TIME interpreted as OWT.")
            # It renames again the matching dictionary to find the respective units. By doing so, it is possible to retrieve units when there are some present, otherwise it returns an empty dictionary
            dict_match = self.read_text(filepath=filepath)
            try:
                for key in dict_match["match_header"]:
                    key_renamed = renamer.rename_key(key)
                    dict_match["match_header"] = [
                        key_renamed if k == key else k
                        for k in dict_match["match_header"]
                    ]
            except Exception:
                print(f"{filepath} - Headers were not renamed")
                pass
            # Now that the header and data is already matched, it will be tried to match the units
            if len(dict_match["match_units"]) > 0:
                units = self.confirm_units(dict_match)
            else:
                units = dict()
            header = list(dict_verified.keys())

            dict_header = {}
            dict_units = {}
            for measure in (
                header
            ):  # It will associate every header to a specific measure in order.
                if measure == "MD":
                    dict_header[measure] = dict_verified[measure]
                    dict_units[measure] = units.get(measure, None)

                elif measure == "TVD":
                    dict_header["TVD"] = dict_verified[measure]
                    dict_units[measure] = units.get(measure, None)

                elif measure == "TVDSS":
                    dict_header["TVDSS"] = dict_verified[measure]
                    dict_units[measure] = units.get(measure, None)

                elif measure == "TWT":
                    dict_header["TWT"] = dict_verified[measure]
                    dict_units[measure] = units.get(measure, None)

                elif measure == "OWT":
                    dict_header["TWT"] = [
                        float(num) * 2 for num in dict_verified[measure]
                    ]
                    qc_description.append(
                        "{} renamed to {}, and values were multiplied by 2.".format(
                            measure, "TWT"
                        )
                    )
                    dict_units["TWT"] = units.get(measure, None)

            if len(dict_header.keys()) >= 2:
                # Data output only happens if there are more than two numeric columns with a header
                headers = dict_header.keys()
                universe = ["MD", "TVD", "TVDSS", "TWT"]
                universe_set = set(universe)

                for head in headers:
                    column = (
                        str(dict_header[head])
                        .replace("[", "")
                        .replace("'", "")
                        .replace("]", "")
                    )
                    # feature.setAttribute(head, column)
                    try:
                        unit_column = head + "_unit"
                        unit = dict_units.get(head, None)
                        if unit is None:
                            unit = "None"
                        # feature.setAttribute(unit_column, unit)
                    except Exception:
                        print(f"{filepath} - No unit was found for {head}")
                        pass
                # -999.25 array is associated to headers without values for each specific file.
                not_in_my_list = list(universe_set.difference(headers))
                for nul in not_in_my_list:
                    nul_number = (
                        str([-999.25 for i in range(0, len(dict_header[head]))])
                        .replace("[", "")
                        .replace("'", "")
                        .replace("]", "")
                    )
                    # feature.setAttribute(nul, nul_number)
                regex = str(dict_match["match_header"])
                # feature.setAttribute("regex", regex)
            else:
                pass
            # outputs the description of the qc
            qc_description = (
                str(qc_description).replace("[", "").replace("'", "").replace("]", "")
            )
            # feature.setAttribute("qc_description", qc_description)
        except Exception as e:
            print(f"{filepath} cannot be read. Reason: {e}")
            pass
        # self.pyoutput(feature)

    def close(self):
        """This method is called once all the FME Features have been processed
        from input().
        """
        pass


from IPython import embed

filepath = "G:\\Sub_Appl_Data\\WellDB\\GB\\wells\\0014\\GB 14-24a- 3\\07.Borehole_Seismic\\14_24a-_3_well_digital_seismic_CSHOT_FILE_236269151.xlsx"
filepath2 = "G:\\Sub_Appl_Data\\WellDB\\GB\\wells\\0014\\GB 14-30a- 4\\07.Borehole_Seismic\\25645af_CGG-Checkshot.xlsx"
filepath3 = "G:\\Sub_Appl_Data\\WellDB\\NO\\wells\\0001\\NO 1-3-10\\07.Borehole_Seismic\\TZV_DEPTH_MD_CHECKSHOT_2008-01-07_1.ASC"
feature_processor = FeatureProcessor()
feature_processor.input(filepath2)
