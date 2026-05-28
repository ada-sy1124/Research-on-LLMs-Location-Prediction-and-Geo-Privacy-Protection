SAM_PROMPT_MAPPING = {
    "Architecture": ["building", "house", "wall", "roof", "fence", "bridge", "tower", "balcony"],
    "Infrastructure": ["utility pole", "street light", "barrier", "sidewalk", "signpost", "pipe", "stairs"],
    "Road Markings": ["road marking", "lane line", "crosswalk", "zebra crossing", "painted arrow"],
    "Signage & Text": ["traffic sign", "billboard", "shop sign", "poster", "banner"],
    "Vegetation": ["Vegetation"],
    "Vehicles": ["car", "bus", "truck", "motorcycle", "bicycle", "boat", "van"],
}



GEO_PROMPT = """You are an advanced geolocation model.

TASK OVERVIEW (IMPORTANT):
This task consists of TWO SEQUENTIAL STAGES that must be completed IN ORDER.

STAGE 1 — GEOLOCATION REASONING (INTERNAL):
First, analyze the image and estimate its geographic coordinates using only visible evidence.
Do this reasoning internally. Do NOT output your internal reasoning.
Output format: 'COORDINATES: <latitude>, <longitude>'

STAGE 2 — EVIDENCE EXTRACTION (OUTPUT):
After determining the coordinates in Stage 1, examine the image again.
Identify ONLY the concrete, physical, visible objects that directly support or justify your predicted location.
Then output the final result using this format: 'REASONING: <structured object list>'

Your final output format (EXACTLY TWO LINES) should be:
Line 1: COORDINATES: <latitude>, <longitude>
Line 2: REASONING: <structured object list>

STRUCTURE OF LINE 2 (MUST FOLLOW EXACTLY):
- Line 2 must start with "REASONING: "
- Format: ClassName: obj1, obj2, obj3; NextClass: obj4, obj5; ...
- Classes must be separated by a semicolon and a space: "; "
- Objects within a class must be separated only by a comma and a space: ", "
- If a class has no relevant objects in the image, don't need to include that class in the output.

REASONING CONTENT RULES (STAGE 2 ONLY):
1. List ONLY the individual, countable, physical objects that SUPPORT the predicted location from Stage 1.
2. Every object must be a concrete visible instance. You MUST include a quantifier (e.g., a number, "a/an", or "some") and at least one descriptive adjective before the noun.
3. Do NOT repeat the same object in more than one class.
4. Use ONLY the following predefined class names:
   "Road Markings", "Signage & Text", "Vehicles", "Architecture", "Vegetation", "Infrastructure"

EXAMPLE OUTPUT:
COORDINATES: 51.5074, -0.1278
REASONING: Signage & Text: 2 street name signs, some parking signs; Road Markings: some lane lines, a zebra crossing; Vehicles: a black bus; Architecture: 3 white houses, a blue house; Vegetation: some trees on the left, a tree in front of the parking sign, a patch of grass; Infrastructure: 2 yellow bollards, a red bollard"""


