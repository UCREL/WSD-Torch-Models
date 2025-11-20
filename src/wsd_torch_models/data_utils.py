from importlib.resources import files
from typing import Any

import yaml


def load_usas_mapper(tags_to_filter_out: set[str] | None) -> dict[str, str]:
    """
    Returns a dictionary of USAS tags and their descriptions.

    NOTE: The USAS tags are loaded from a YAML file that can be found in the
    `wsd_torch_models/data/usas/usas_mapper.yaml` file.

    Args:
        tags_to_filter_out (set[str] | None): A set of USAS tags to filter out.

    Returns:
        dict[str, str]: A dictionary of USAS tags and their descriptions.
    """

    def _get_usas_tag_descriptions(usas_tag_name: str,
                                   usas_tag_dict: dict[str, Any],
                                   collected_tag_descriptions: dict[str, str]
                                   ) -> dict[str, str]:
        """
        A recursive function that loops through the `usas_tag_dict` and returns a
        dictionary of the USAS tag and as a value it's description. Each USAS tag
        that is found is added to the `collected_tag_descriptions` dictionary.
        Once all the USAS tags are found, the `collected_tag_descriptions` is
        returned.

        The description is made of the USAS tag title and description in the
        following format: `title: <title> description: <description>`

        Args:
            usas_tag_name (str): The name of the USAS tag.
            usas_tag_dict (dict[str, Any]): A dictionary containing the raw
                USAS tag data that is read from the YAML file.
            collected_tag_descriptions (dict[str, str]): A dictionary of all
                USAS tags and their descriptions.

        Returns:
            dict[str, str]: A dictionary of USAS tags and their descriptions.
        """
        if "title" in usas_tag_dict and "description" in usas_tag_dict:
            title_description = f"title: {usas_tag_dict['title']} description: {usas_tag_dict['description']}"
            if usas_tag_name in collected_tag_descriptions:
                raise KeyError(f"Duplicate usas tag name found: {usas_tag_name} "
                               "when reading the following data: "
                               f"{usas_tag_dict}, currently found usas tags: "
                               f"{collected_tag_descriptions}")
            collected_tag_descriptions[usas_tag_name] = title_description.strip()
        elif "title" in usas_tag_dict:
            raise KeyError("No description key found when it is expected for: "
                           f"{usas_tag_name} {usas_tag_dict}")
        elif "description" in usas_tag_dict:
            raise KeyError("No title key found when it is expected for: "
                           f"{usas_tag_name} {usas_tag_dict}")

        keys_to_ignore = set(["title", "description"])
        for child_usas_tag_name, child_usas_tag_dict in usas_tag_dict.items():
            if child_usas_tag_name not in keys_to_ignore:
                collected_tag_descriptions = _get_usas_tag_descriptions(child_usas_tag_name,
                                                                        child_usas_tag_dict,
                                                                        collected_tag_descriptions)
        return collected_tag_descriptions

    usas_tag_descriptions_file = files("wsd_torch_models") / "data" / "usas" / "usas_mapper.yaml"
    usas_mapping: dict[str, str] = {}
    with usas_tag_descriptions_file.open("r") as usas_mapper_fp:
        usas_mapping_data = usas_mapper_fp.read()
        for high_level_usas_tag, high_level_usas_tag_dict in yaml.safe_load(usas_mapping_data).items():
            usas_mapping = _get_usas_tag_descriptions(high_level_usas_tag,
                                                      high_level_usas_tag_dict,
                                                      usas_mapping)
    if tags_to_filter_out:
        tmp_usas_mapping = {}
        for key, value in usas_mapping.items():
            if key in tags_to_filter_out:
                continue
            tmp_usas_mapping[key] = value
        usas_mapping = tmp_usas_mapping
    return usas_mapping
