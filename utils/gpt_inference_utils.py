import openai
import base64
import json

ADDITIONAL_MATERIAL_PROMPT = "In each caption, you will also be provided the most likely material of the object. It will be delimited with brackets. You should trust this material and not give out materials that does not appear with it."
PRED_CAND_MATS_DENSITY_SYS_MSG_ENHANCED = f"""You will be provided with captions that each describe an image of an object. The captions will be delimited with quotes ("). {ADDITIONAL_MATERIAL_PROMPT} Based on the caption, give me 5 materials that the object might be made of, along with the mass densities (in kg/m^3) of each of those materials. You may provide a range of values for the mass density instead of a single value. Try to consider all the possible parts of the object. Do not include coatings like "paint" in your answer.

Format Requirement:
You must provide your answer as a list of 5 (material: mass density) pairs, each separated by a semi-colon (;). Do not include any other text in your answer, as it will be parsed by a code script later. Your answer must look like:
(material 1: low-high kg/m^3);(material 2: low-high kg/m^3);(material 3: low-high kg/m^3);(material 4: low-high kg/m^3);(material 5: low-high kg/m^3)
"""

PRED_CAND_MATS_DENSITY_SYS_MSG = f"""You will be provided with captions that each describe an image of an object. The captions will be delimited with quotes ("). Based on the caption, give me 5 materials that the object might be made of, along with the mass densities (in kg/m^3) of each of those materials. You may provide a range of values for the mass density instead of a single value. Try to consider all the possible parts of the object. Do not include coatings like "paint" in your answer.

Format Requirement:
You must provide your answer as a list of 5 (material: mass density) pairs, each separated by a semi-colon (;). Do not include any other text in your answer, as it will be parsed by a code script later. Your answer must look like:
(material 1: low-high kg/m^3);(material 2: low-high kg/m^3);(material 3: low-high kg/m^3);(material 4: low-high kg/m^3);(material 5: low-high kg/m^3)
"""

PRED_CAND_MATS_DENSITY_SYS_MSG_4V = """You will be given an image of an object. Based on the image, give me a short (5-10 words) description of what the object is, and also 5 materials (e.g. wood, plastic, foam) that the object might be made of, along with the mass densities (in kg/m^3) of each of those materials. You may provide a range of values for the mass density instead of a single value. Try to consider all the possible parts of the object. The material you predict should be a single word, not a phrase. Do not include coatings like "paint" in your answer.

Format Requirement:
You must provide your answer in the following JSON format, as it will be parsed by a code script later. Your answer must look like:
{
    "description": description
    "materials": [
        {"name": material1, "mass density (kg/m^3)": "low-high"},
        {"name": material2, "mass density (kg/m^3)": "low-high"},
        {"name": material3, "mass density (kg/m^3)": "low-high"},
        {"name": material4, "mass density (kg/m^3)": "low-high"},
        {"name": material5, "mass density (kg/m^3)": "low-high"}
    ]
}
Do not include any other text in your answer. Do not include unnecessary words besides the material in the material name. 
"""

PRED_CAND_MATS_DENSITY_SYS_MSG_4V_ENHANCED = """You will be given an image of an object. Based on the image, you should analyze it step by step.
Firstly give me a short (5-10 words) description of what the object is. 
Then you need to consider what sub-parts this object is made up of. For each sub-part, give me one material (e.g. wood, plastic, foam) that this part might be made of, along with the mass densities (in kg/m^3) of this material. You may provide a range of values for the mass density instead of a single value. The material you predict should be a single word, not a phrase. Do not include coatings like "paint" in your answer.
Finally, you need to consider your description of the object and predict the pure volume(in m^3). Pure volume is the sum of the volumes of all solid areas of the object, excluding empty areas. When predicting pure volume, you can imagine flattening all the components of an object and stacking them tightly together to obtain pure volume.
Note that the same object should have the same pure volume in different states (e.g. open and closed). For objects like bed and sofa, they are usually made up of some frames inside, and pure volume should only consider the volume of these frames, and should not consider other spatial areas. For example, the pure volume of a bed is usually 0.02 m^3, the pure volume of a sofa is also close to 0.02 m^3, and the pure volume of a pillow is close to 0.004 m^3. For objects like bookcases, drawers, and containers, please treat them as empty, which do not contain any other objects. For each object, please consider which are possible hollow places, and then predict the pure volume.

Format Requirement:
You must provide your answer in the following JSON format, as it will be parsed by a code script later. Your answer must look like:
{
    "description": description,
    "materials": [
        {"name": material1, "mass density (kg/m^3)": "low-high"},
        {"name": material2, "mass density (kg/m^3)": "low-high"},
        {"name": material3, "mass density (kg/m^3)": "low-high"},
        {"name": material4, "mass density (kg/m^3)": "low-high"},
        {"name": material5, "mass density (kg/m^3)": "low-high"}
    ],
    "pure_volume": pure volume
}
Do not include any other text in your answer. Do not include unnecessary words besides the material in the material name. 
"""

PRED_CAND_MATS_HARDNESS_SYS_MSG = """You will be provided with captions that each describe an image of an object. The captions will be delimited with quotes ("). Based on the caption, give me 3 materials that the object might be made of, along with the hardness of each of those materials. Choose whether to use Shore A hardness or Shore D hardness depending on the material. You may provide a range of values for hardness instead of a single value. Try to consider all the possible parts of the object.pu

Format Requirement:
You must provide your answer as a list of 3 (material: hardness, Shore A/D) tuples, each separated by a semi-colon (;). Do not include any other text in your answer, as it will be parsed by a code script later. Your answer must look like:
(material 1: low-high, <Shore A or Shore D>);(material 2: low-high, <Shore A or Shore D>);(material 3: low-high, <Shore A or Shore D>)
Make sure to use Shore A or Shore D hardness, not Mohs hardness.
"""

PRED_CAND_MATS_FRICTION_SYS_MSG = """You will be provided with captions that each describe an image. The captions will be delimited with quotes ("). Based on the caption, give me 3 materials that the surfaces in the image might be made of, along with the kinetic friction coefficient of each material when sliding against a fabric surface. You may provide a range of values for the friction coefficient instead of a single value. Try to consider all the possible surfaces.

Format Requirement:
You must provide your answer as a list of 3 (material: friction coefficient) pairs, each separated by a semi-colon (;). Do not include any other text in your answer, as it will be parsed by a code script later. Your answer must look like:
(material 1: low-high);(material 2: low-high);(material 3: low-high)
Try to provide as narrow of a range as possible for the friction coefficient.
"""

PRED_THICKNESS_SYS_MSG = """You will be provided with captions that each describe an image of an object, along with a set of possible materials used to make the object. For each material, estimate the thickness (in cm) of that material in the object. You may provide a range of values for the thickness instead of a single value.

Format Requirement:
You must provide your answer as a list of 5 (material: thickness) pairs, each separated by a semi-colon (;). Do not include any other text in your answer, as it will be parsed by a code script later. Your answer must look like:
(material 1: low-high cm);(material 2: low-high cm);(material 3: low-high cm);(material 4: low-high cm);(material 5: low-high cm)
"""

PRED_CAND_MATS_YOUNGsMODULUS_SYS_MSG = f"""You will be provided with captions that each describe an image of an object. The captions will be delimited with quotes ("). Based on the caption, give me 5 materials that the object might be made of, along with the Young's modulus (in GPa) of each of those materials. You may provide a range of values for the Young's modulus instead of a single value. Try to consider all the possible parts of the object. Do not include coatings like "paint" in your answer.

Format Requirement:
You must provide your answer as a list of 5 (material: Young's modulus) pairs, each separated by a semi-colon (;). Do not include any other text in your answer, as it will be parsed by a code script later. Your answer must look like:
(material 1: low-high GPa);(material 2: low-high GPa);(material 3: low-high GPa);(material 4: low-high GPa);(material 5: low-high GPa)
"""

PRED_THICKNESS_EXAMPLE_INPUT_1 = 'Caption: "a lamp with a white shade" Materials: "fabric, plastic, metal, ceramic, glass"'
PRED_THICKNESS_EXAMPLE_OUTPUT_1 = "(fabric: 0.1-0.2 cm);(plastic: 0.3-1.0 cm);(metal: 0.1-0.2 cm);(ceramic: 0.2-0.5 cm);(glass: 0.3-0.8 cm)"
PRED_THICKNESS_EXAMPLE_INPUT_2 = 'Caption: "a grey ottoman" Materials: "wood, fabric, foam, metal, plastic"'
PRED_THICKNESS_EXAMPLE_OUTPUT_2 = "(wood: 2.0-4.0 cm);(fabric: 0.2-0.5 cm);(foam: 5.0-15.0 cm);(metal: 0.1-0.2 cm);(plastic: 0.5-1.0 cm)"
PRED_THICKNESS_EXAMPLE_INPUT_3 = 'Caption: "a white frame" Materials: "plastic, wood, aluminum, steel, glass"'
PRED_THICKNESS_EXAMPLE_OUTPUT_3 = "(plastic: 0.1-0.3 cm);(wood: 1.0-1.5 cm);(aluminum: 0.1-0.3 cm);(steel: 0.1-0.2 cm);(glass: 0.2-0.5 cm)"
PRED_THICKNESS_EXAMPLE_INPUT_4 = 'Caption: "a metal rack with three shelves" Materials: "steel, aluminum, wood, plastic, iron"'
PRED_THICKNESS_EXAMPLE_OUTPUT_4 = "(steel: 0.1-0.2 cm);(aluminum: 0.1-0.3 cm);(wood: 1.0-2.0 cm);(plastic: 0.5-1.0 cm);(iron: 0.5-1.0 cm)"


def gpt_candidate_materials(caption, property_name='density', model_name='gpt-4o-mini', seed=100, enhanced=False):
    if property_name == 'density':
        if enhanced:
            sys_msg = PRED_CAND_MATS_DENSITY_SYS_MSG_ENHANCED
        else:
            sys_msg = PRED_CAND_MATS_DENSITY_SYS_MSG
    elif property_name == 'hardness':
        sys_msg = PRED_CAND_MATS_HARDNESS_SYS_MSG
    elif property_name == 'friction':
        sys_msg = PRED_CAND_MATS_FRICTION_SYS_MSG
    elif property_name == 'youngs-modulus':
        sys_msg = PRED_CAND_MATS_YOUNGsMODULUS_SYS_MSG
    else:
        raise NotImplementedError
    response = openai.ChatCompletion.create(
      model=model_name,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": '"%s"' % caption},
        ],
        request_timeout=20,
        seed=seed,
    )
    return response['choices'][0]['message']['content']


def gpt_thickness(caption, candidate_materials, mode='list', model_name='gpt-4o-mini', seed=100):
    if mode == 'list':
        mat_names, mat_vals = parse_material_list(candidate_materials)
    elif mode == 'json':
        caption, mat_names, mat_vals = parse_material_json(candidate_materials)
    else:
        raise NotImplementedError
    mat_names_str = ', '.join(mat_names)
    user_msg = 'Caption: "%s" Materials: "%s"' % (caption, mat_names_str)

    response = openai.ChatCompletion.create(
      model=model_name,
        messages=[
            {"role": "system", "content": PRED_THICKNESS_SYS_MSG},
            {"role": "user", "content": PRED_THICKNESS_EXAMPLE_INPUT_1},
            {"role": "assistant", "content": PRED_THICKNESS_EXAMPLE_OUTPUT_1},
            {"role": "user", "content": PRED_THICKNESS_EXAMPLE_INPUT_2},
            {"role": "assistant", "content": PRED_THICKNESS_EXAMPLE_OUTPUT_2},
            {"role": "user", "content": PRED_THICKNESS_EXAMPLE_INPUT_3},
            {"role": "assistant", "content": PRED_THICKNESS_EXAMPLE_OUTPUT_3},
            {"role": "user", "content": PRED_THICKNESS_EXAMPLE_INPUT_4},
            {"role": "assistant", "content": PRED_THICKNESS_EXAMPLE_OUTPUT_4},
            {"role": "user", "content": user_msg},
        ],
        request_timeout=20,
        seed=seed,
    )
    return response['choices'][0]['message']['content']


def parse_material_list(matlist, max_n=5):
    elems = matlist.split(';')
    if len(elems) > max_n:
        print('too many materials %s' % matlist)
        return None
    
    mat_names = []
    mat_vals = []

    for elem in elems:
        elem_parts = elem.strip().split(':')
        if len(elem_parts) != 2: 
            print('bad format %s' % matlist)
            return None
        mat_name, values = elem_parts
        if not mat_name.startswith('(') or mat_name[1].isnumeric() or mat_name.startswith('(material 1'):
            print('bad format %s' % matlist)
            return None

        mat_name = mat_name[1:]
        mat_names.append(mat_name.lower())  # force lowercase

        values = values.strip().split(' ')[0]
        values = values.replace(",", "")
        if values[-1] == ')':
            values = values[:-1]

        # Value may or may not be a range
        splitted = values.split('-')
        try:
            float(splitted[0])
        except ValueError:
            print('value cannot be converted to float %s' % matlist)
            return None
        if len(splitted) == 2:
            mat_vals.append([float(splitted[0]), float(splitted[1])])
        elif len(splitted) == 1:
            mat_vals.append([float(splitted[0]), float(splitted[0])])
        else:
            print('bad format %s' % matlist)
            return None
        
    return mat_names, mat_vals


def parse_material_hardness(matlist, max_n=5):
    elems = matlist.split(';')
    if len(elems) > max_n:
        print('too many materials %s' % matlist)
        return None
    
    mat_names = []
    mat_vals = []

    for elem in elems:
        elem_parts = elem.strip().split(':')
        if len(elem_parts) != 2: 
            print('bad format %s' % matlist)
            return None
        mat_name, values = elem_parts
        if not mat_name.startswith('(') or mat_name[1].isnumeric() or mat_name.startswith('(material 1'):
            print('bad name %s' % matlist)
            return None

        mat_name = mat_name[1:]
        mat_names.append(mat_name.lower())  # force lowercase

        values = values.strip().split(',')
        units = values[-1].split(' ')[-1][:-1]
        if units not in ['A', 'D']:
            print('bad units %s' % matlist)
            return None
        values = values[0]
        values = values.replace(",", "")

        # Value may or may not be a range
        splitted = values.split('-')
        try:
            float(splitted[0])
        except ValueError:
            print('value cannot be converted to float %s' % matlist)
            return None
        if len(splitted) == 2:
            mat_vals.append([float(splitted[0]), float(splitted[1])])
        elif len(splitted) == 1:
            mat_vals.append([float(splitted[0]), float(splitted[0])])
        else:
            print('bad format %s' % matlist)
            return None
        
        if units == 'D':
            mat_vals[-1][0] += 100
            mat_vals[-1][1] += 100
        
    return mat_names, mat_vals


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def gpt4v_candidate_materials(image_path, property_name='density', seed=100, model_name='gpt-4-turbo'):
    if property_name == 'density':
        sys_msg = PRED_CAND_MATS_DENSITY_SYS_MSG_4V_ENHANCED
    else:
        raise NotImplementedError
    
    base64_image = encode_image(image_path)

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
        {
            "role": "system",
            "content": sys_msg
        },
        {
            "role": "user",
            "content": [
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
                },
            ]
        }
        ],
        request_timeout=30,
        max_tokens=300,
        seed=seed,
    )
    return response['choices'][0]['message']['content']


def parse_material_json(matjson, max_n=5, field_name='mass density (kg/m^3)'):
    desc_and_mats = json.loads(matjson)
    if 'description' not in desc_and_mats or 'materials' not in desc_and_mats or 'pure_volume' not in desc_and_mats:
        print('bad format %s' % matjson)
        return None
    mat_names = []
    mat_vals = []
    for mat in desc_and_mats['materials']:
        if 'name' not in mat or field_name not in mat:
            print('bad format %s' % matjson)
            return None
        mat_name = mat['name']
        mat_names.append(mat_name.lower())  # force lowercase
        values = mat[field_name]
        # Value may or may not be a range
        splitted = values.split('-')
        try:
            float(splitted[0])
        except ValueError:
            print('value cannot be converted to float %s' % matjson)
            return None
        if len(splitted) == 2:
            mat_vals.append([float(splitted[0]), float(splitted[1])])
        elif len(splitted) == 1:
            mat_vals.append([float(splitted[0]), float(splitted[0])])
        else:
            print('bad format %s' % matjson)
            return None
    return desc_and_mats['description'], mat_names, mat_vals, desc_and_mats['pure_volume']