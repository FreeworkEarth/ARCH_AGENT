"""
M-Score calculation exactly as implemented in DV8 MScoreService.java and the paper
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict





def load_json(path: Path) -> dict:
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)





def parse_drh_structure(drh_data: dict) -> List[List[List[str]]]:
    """
    Parse DRH JSON into the structure DV8 uses:
    List<List<List<String>>> = layers -> modules -> files

    Exactly matches splitModules() in MScoreService.java (lines 44-68)
    """
    layers = []

    # DV8 DRH structure has layers as top-level groups
    if 'structure' in drh_data:
        for layer_item in drh_data['structure']:
            if layer_item.get('@type') == 'group':  # This is a layer
                layer_modules = []

                # Each layer contains modules (which can be groups or items)
                if 'nested' in layer_item:
                    for module_item in layer_item['nested']:
                        if module_item.get('@type') == 'group':
                            # This is a module (group) - collect all file names recursively
                            files = collect_all_items(module_item)
                            if files:
                                layer_modules.append(files)
                        elif module_item.get('@type') == 'item':
                            # Single file module
                            layer_modules.append([module_item['name']])

                if layer_modules:
                    layers.append(layer_modules)

    return layers






def collect_all_items(group: dict) -> List[str]:
    """Recursively collect all item names from a group (module)"""
    items = []

    if 'nested' in group:
        for item in group['nested']:
            if item.get('@type') == 'item':
                items.append(item['name'])
            elif item.get('@type') == 'group':
                # Recursively collect from subgroups
                items.extend(collect_all_items(item))

    return items





def calculate_mscore_dv8_exact(dsm_path: Path, drh_path: Path, file_filter: List[str] = None, include_details: bool = False) -> Dict:
    """
    Calculate M-Score exactly as DV8 does it in MScoreService.java

    Matches calculateMetric() method (lines 70-212)

    Args:
        dsm_path: Path to DSM JSON file
        drh_path: Path to DRH JSON file
        file_filter: Optional list of filenames to include (for split analysis)
                     If None, uses all files from DRH structure
    """
    # Load data
    dsm_data = load_json(dsm_path)
    drh_data = load_json(drh_path)

    # Get matrix variables and build adjacency matrix
    matrix_variables = dsm_data.get('variables', [])
    cells = dsm_data.get('cells', [])

    # Build filename to matrix index map (lines 75-82)
    filename_to_matrix_index = {}
    for index, filename in enumerate(matrix_variables):
        filename_to_matrix_index[filename] = index

    # Build adjacency matrix (check if cell has dependencies)
    adj_matrix = {}
    for cell in cells:
        src_idx = cell['src']
        dest_idx = cell['dest']
        values = cell.get('values', [])
        if values:  # Has dependencies
            adj_matrix[(src_idx, dest_idx)] = True

    # Parse DRH structure (line 38)
    cluster_as_lists = parse_drh_structure(drh_data)

    if not cluster_as_lists:
        return {'mscore': 0.0, 'mscore_percentage': 0.0}

    # Lines 73-109: Create maps and calculate sizes
    filename_to_layer_module_string = {}
    all_files_to_process = []
    calculated_total_size = 0
    size_per_module_map = {}
    module_files_map = {}
    layer_size_map = {}

    for layer_num in range(len(cluster_as_lists)):
        modules_as_lists = cluster_as_lists[layer_num]
        size_for_current_layer = 0

        for module_num in range(len(modules_as_lists)):
            key_string = f"{layer_num},{module_num}"
            files_as_list = modules_as_lists[module_num]
            current_module_size = len(files_as_list)

            size_per_module_map[key_string] = current_module_size
            module_files_map[key_string] = list(files_as_list)
            calculated_total_size += current_module_size
            size_for_current_layer += current_module_size

            for file in files_as_list:
                filename_to_layer_module_string[file] = key_string
                all_files_to_process.append(file)

        layer_size_map[layer_num] = size_for_current_layer

    # Apply file filter if provided (for split analysis)
    if file_filter is not None:
        file_filter_set = set(file_filter)

        # Filter all_files_to_process
        filtered_files = [f for f in all_files_to_process if f in file_filter_set]
        all_files_to_process = filtered_files

        # Rebuild size maps based on filtered files
        filename_to_layer_module_string_filtered = {}
        size_per_module_map_filtered = {}
        layer_size_map_filtered = {}
        calculated_total_size = 0

        for layer_num in range(len(cluster_as_lists)):
            modules_as_lists = cluster_as_lists[layer_num]
            size_for_current_layer = 0

            for module_num in range(len(modules_as_lists)):
                key_string = f"{layer_num},{module_num}"
                files_as_list = modules_as_lists[module_num]

                # Filter files in this module
                filtered_module_files = [f for f in files_as_list if f in file_filter_set]
                current_module_size = len(filtered_module_files)

                if current_module_size > 0:
                    size_per_module_map_filtered[key_string] = current_module_size
                    calculated_total_size += current_module_size
                    size_for_current_layer += current_module_size

                    for file in filtered_module_files:
                        filename_to_layer_module_string_filtered[file] = key_string

            if size_for_current_layer > 0:
                layer_size_map_filtered[layer_num] = size_for_current_layer

        # Replace with filtered versions
        filename_to_layer_module_string = filename_to_layer_module_string_filtered
        size_per_module_map = size_per_module_map_filtered
        layer_size_map = layer_size_map_filtered

    # Lines 110-125: Calculate lower and upper level maps
    lower_level_map = {}
    upper_level_map = {}

    existing_layers = sorted(layer_size_map.keys())
    if not existing_layers:
        return {'mscore': 0.0, 'mscore_percentage': 0.0}

    first_layer = existing_layers[0]
    lower_level_map[first_layer] = calculated_total_size - layer_size_map[first_layer]
    upper_level_map[first_layer] = 0

    for idx in range(1, len(existing_layers)):
        layer_num = existing_layers[idx]
        prev_layer = existing_layers[idx - 1]

        previous_lower_level_layer = lower_level_map[prev_layer]
        current_layer_size = layer_size_map[layer_num]
        lower_level_value = previous_lower_level_layer - current_layer_size
        lower_level_map[layer_num] = lower_level_value

        previous_upper_level_layer = upper_level_map[prev_layer]
        previous_layer_size = layer_size_map[prev_layer]
        upper_level_value = previous_upper_level_layer + previous_layer_size
        upper_level_map[layer_num] = upper_level_value

    # Lines 127-170: Calculate dependencies
    dest_files_to_set_of_dependencies = {}
    layer_module_key_to_dest_number_of_deps = {}
    layer_module_key_to_source_number_of_deps = {}
    layer_module_key_to_internal_dependency_count = {}

    for source_file in all_files_to_process:
        # Line 134: Java assumes file exists in map (no null check in original)
        if source_file not in filename_to_matrix_index:
            continue
        source_index = filename_to_matrix_index[source_file]

        for destination_file in all_files_to_process:
            # Line 137: Java assumes file exists in map
            if destination_file not in filename_to_matrix_index:
                continue
            destination_index = filename_to_matrix_index[destination_file]

            # Line 138: Check if there's a dependency
            if (source_index, destination_index) in adj_matrix:
                source_module_string = filename_to_layer_module_string[source_file]
                dest_module_string = filename_to_layer_module_string[destination_file]

                # Line 143: isInSameModule check (== is reference equality in Java, content equality in Python)
                is_in_same_module = (source_module_string == dest_module_string)

                if not is_in_same_module:
                    # Lines 146-148: Use Set to avoid double counting
                    if dest_module_string not in dest_files_to_set_of_dependencies:
                        dest_files_to_set_of_dependencies[dest_module_string] = set()
                    dest_files_to_set_of_dependencies[dest_module_string].add(source_file)

                    # Line 150: isConformingToDRH - CASE INSENSITIVE comparison
                    # Python doesn't have compareToIgnoreCase, so convert to lower
                    is_conforming_to_drh = (dest_module_string.lower() < source_module_string.lower())

                    if is_conforming_to_drh:  # Lines 151-160
                        dest_dep_number = layer_module_key_to_dest_number_of_deps.get(dest_module_string, 0)
                        dest_dep_number += 1
                        layer_module_key_to_dest_number_of_deps[dest_module_string] = dest_dep_number

                        source_dep_number = layer_module_key_to_source_number_of_deps.get(source_module_string, 0)
                        source_dep_number += 1
                        layer_module_key_to_source_number_of_deps[source_module_string] = source_dep_number
                else:
                    # Lines 162-167: Internal dependencies
                    internal_count = layer_module_key_to_internal_dependency_count.get(dest_module_string, 0)
                    internal_count += 1
                    layer_module_key_to_internal_dependency_count[dest_module_string] = internal_count

    # Lines 174-209: Calculate M-Score
    MODULESIZE_THRESHOLD = 5
    MINIMUM_MODULE_SIZE_FOR_CONNECTED_PENALTY_TO_APPLY = 5

    m_score_value = 0.0
    module_details = [] if include_details else None
    size_factor_sum = 0.0
    clddf_sum = 0.0
    imcf_sum = 0.0
    component_count = 0

    for layer_number in range(len(cluster_as_lists)):
        # Skip layers that don't exist in filtered data
        if layer_number not in layer_size_map:
            continue

        modules = cluster_as_lists[layer_number]

        for module_number in range(len(modules)):
            key_string = f"{layer_number},{module_number}"

            # Skip modules that don't exist in filtered data
            if key_string not in size_per_module_map:
                continue

            current_module_size = size_per_module_map[key_string]

            # Line 182: Size Factor
            size_factor_value = current_module_size / calculated_total_size

            # Lines 183-187: Size Penalty
            size_penalty_value = 1.0
            if current_module_size > MODULESIZE_THRESHOLD:
                size_penalty_value = 1.0 / (math.log(current_module_size) / math.log(MODULESIZE_THRESHOLD))

            # Lines 189-195: Cross-layer Dependency Penalty
            dep_penalty_value_lower_level = 0.0
            if layer_number in lower_level_map:
                lower_level_files = lower_level_map[layer_number]
                if lower_level_files != 0:
                    dest_deps_for_module = layer_module_key_to_dest_number_of_deps.get(key_string, 0)
                    dep_penalty_value_lower_level = dest_deps_for_module / (lower_level_files * current_module_size)

            # Lines 197-204: Internal Complexity Penalty
            internal_edges = layer_module_key_to_internal_dependency_count.get(key_string, 0)
            number_of_nodes = current_module_size
            connected_penalty = 0.0
            if number_of_nodes > MINIMUM_MODULE_SIZE_FOR_CONNECTED_PENALTY_TO_APPLY:
                connected_penalty = internal_edges / (number_of_nodes * (number_of_nodes - 1))

            # Line 206: Module M-Score
            current_module_mscore_value = (size_factor_value * size_penalty_value *
                                          (1.0 - dep_penalty_value_lower_level) *
                                          (1.0 - connected_penalty))

            # Line 207: Add to total
            m_score_value += current_module_mscore_value

            # Accumulate rollups
            size_factor_sum += size_factor_value
            clddf_sum += (1.0 - dep_penalty_value_lower_level)
            imcf_sum += (1.0 - connected_penalty)
            component_count += 1

            if include_details:
                module_details.append({
                    'layer': layer_number,
                    'module': module_number,
                    'module_key': key_string,
                    'module_size': current_module_size,
                    'files': module_files_map.get(key_string, []),
                    'size_factor': size_factor_value,
                    'size_penalty': size_penalty_value,
                    'cross_penalty': dep_penalty_value_lower_level,
                    'internal_penalty': connected_penalty,
                    'clddf': 1.0 - dep_penalty_value_lower_level,
                    'imcf': 1.0 - connected_penalty,
                    'contribution': current_module_mscore_value,
                })

    # Line 211: Return value (DV8 returns as decimal 0-1, we also return as percentage)
    result = {
        'mscore': m_score_value,
        'mscore_percentage': m_score_value * 100,
        'total_files': calculated_total_size,
        'num_layers': len(cluster_as_lists),
        'num_modules': sum(len(modules) for modules in cluster_as_lists),
        'module_details': module_details
    }

    if include_details and component_count > 0:
        result['component_rollup'] = {
            'size_factor_total': size_factor_sum,
            'avg_clddf': clddf_sum / component_count,
            'avg_imcf': imcf_sum / component_count,
            'module_count': component_count
        }

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python mscore_dv8_exact.py <dsm_path> <drh_path>")
        sys.exit(1)

    dsm_path = Path(sys.argv[1])
    drh_path = Path(sys.argv[2])

    result = calculate_mscore_dv8_exact(dsm_path, drh_path)

    print(f"\nM-Score (DV8 Exact Implementation):")
    print(f"  M-Score: {result['mscore_percentage']:.2f}%")
    print(f"  Total files: {result['total_files']}")
    print(f"  Layers: {result['num_layers']}")
    print(f"  Modules: {result['num_modules']}")
