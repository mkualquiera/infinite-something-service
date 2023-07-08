import argparse
import base64
import io
import json
import os
import tempfile
import time
import typing as tp
from typing import Any, Optional, List, Dict

import torch
import torchaudio
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rich.console import Console
from tqdm import trange
import json
import os
from typing import Optional

from audiocraft.models import MusicGen
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh
from io import StringIO
import numpy as np

import transformers
import dotenv
import openai


# ==  Game inference ==

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


app = FastAPI()
console = Console()

# # create static route for serving test html
# app.mount("/static", StaticFiles(directory="static"), name="static")


# == Texture generation ==
def patch_conv(**patch):
    cls = torch.nn.Conv2d
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, **patch)

    cls.__orig_init__ = init
    cls.__init__ = __init__


def unpatch_conv():
    cls = torch.nn.Conv2d
    cls.__init__ = cls.__orig_init__


patch_conv(padding_mode="circular")
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    custom_pipeline="sd_text2img_k_diffusion",
)
unpatch_conv()
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")


class TextureInferenceRequest(BaseModel):
    text: str
    sampler: str = "sample_dpmpp_2m"


class TextureInferenceResponse(BaseModel):
    image: str


app.pipe_lock = False


@app.post("/generate_texture")
@torch.no_grad()
def generate(request: TextureInferenceRequest) -> TextureInferenceResponse:
    while app.pipe_lock:
        time.sleep(0.1)
    app.pipe_lock = True
    console.print(request)
    t = time.perf_counter()
    pipe.set_sampler(request.sampler)
    image = pipe(request.text).images[0]
    console.print(f"Generated image in", time.perf_counter() - t, "seconds")

    # Convert PIL image to bytes
    bytes = io.BytesIO()
    image.save(bytes, format="PNG")
    image = bytes.getvalue()

    # Convert to b64
    image = base64.b64encode(image)
    image = image.decode("utf-8")
    app.pipe_lock = False

    return TextureInferenceResponse(image=image)


# == Music generation ==
music_model = MusicGen.get_pretrained("small", device="cuda")


class MusicInferenceRequest(BaseModel):
    text: str
    duration: float = 30
    loops: int = 4


class MusicInferenceResponse(BaseModel):
    audio: str


@app.post("/generate_music")
@torch.no_grad()
def generate(request: MusicInferenceRequest) -> MusicInferenceResponse:
    console.print(request)
    console.print("Sample rate:", music_model.sample_rate)
    t = time.perf_counter()
    attributes, prompt_tokens = music_model._prepare_tokens_and_attributes(
        [request.text], None
    )

    music_model.generation_params = {
        "max_gen_len": int(request.duration * music_model.frame_rate),
        "use_sampling": True,
        "temp": 1.0,
        "top_k": 250,
        "top_p": 0,
        "cfg_coef": 3.0,
        "two_step_cfg": 0,
    }
    total = []
    for _ in trange(request.loops):
        with music_model.autocast:
            gen_tokens = music_model.lm.generate(
                prompt_tokens,
                attributes,
                callback=None,
                **music_model.generation_params,
            )
            total.append(
                gen_tokens[
                    ..., prompt_tokens.shape[-1] if prompt_tokens is not None else 0 :
                ]
            )
            prompt_tokens = gen_tokens[..., -gen_tokens.shape[-1] // 2 :]
    gen_tokens = torch.cat(total, -1)

    assert gen_tokens.dim() == 3
    console.print("gen_tokens information")
    console.print("Shape:", gen_tokens.shape)
    console.print("Dtype:", gen_tokens.dtype)
    console.print("Contents:", gen_tokens)
    with torch.no_grad():
        gen_audio = music_model.compression_model.decode(gen_tokens, None)
    console.print("gen_audio information")
    console.print("Shape:", gen_audio.shape)
    console.print("Dtype:", gen_audio.dtype)
    console.print("Contents:", gen_audio)
    gen_audio = gen_audio.cpu()

    # Save to tempfile
    # with tempfile.NamedTemporaryFile("rb", suffix=".wav") as f:
    #    # audio_write(f.name, gen_audio, music_model.frame_rate)
    #    torchaudio.save(f.name, gen_audio[0], music_model.sample_rate)
    #    # Read bytes from tempfile
    #    f.seek(0)
    #    audio = f.read()

    # convert to signed 8 bit signed int
    gen_audio = gen_audio * 128
    gen_audio = gen_audio.to(torch.int8)

    console.print("gen_audio information")
    console.print("Shape:", gen_audio.shape)
    console.print("Dtype:", gen_audio.dtype)
    console.print("Contents:", gen_audio)

    # to bytes
    audio = gen_audio.numpy().tobytes()

    console.print(f"Generated audio in", time.perf_counter() - t, "seconds")

    # Convert to b64
    audio = base64.b64encode(audio)
    audio = audio.decode("utf-8")

    return MusicInferenceResponse(audio=audio)


# == World model ==
class OpenAILLMInteface(object):
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    def __call__(self, prompt: List[Dict[str, str]]) -> str:
        return (
            openai.ChatCompletion.create(model=self.model_name, messages=prompt)
            .choices[0]
            .message.content
        )


class HfLLMInterface(object):
    def __init__(self, model_name, load_kwargs={}) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, prompt: List[Dict[str, str]]) -> Any:
        text = []
        for part in prompt:
            if part["role"] == "user":
                text.append(f"User: {part['content']}")
            elif part["role"] == "assistant":
                text.append(f"Assistant: {part['assistant']}")
            elif part["role"] == "system":
                text.append(part["content"])
        text.append("Assistant:")
        text = "\n".join(text)
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        with torch.autocast("cuda"):
            newline = self.tokenizer.encode("\n")[0]
            output_ids = self.model.generate(
                input_ids,
                top_k=40,
                top_p=0.95,
                repetition_penalty=1.1,
                max_new_tokens=64,
                stopping_criteria=[
                    lambda ids, *_: "\n" in self.tokenizer.decode(ids[0][-1])
                ],
            )[0, input_ids.shape[1] :]
            return self.tokenizer.decode(output_ids)


if os.environ.get("USE_OAI", "0") == "1":
    # print red bold
    console.print("[bold red]Using OpenAI API[/bold red]")
    llms = {
        "tiny": OpenAILLMInteface("gpt-3.5-turbo"),
        "regular": OpenAILLMInteface("gpt-3.5-turbo"),
        "hq": OpenAILLMInteface("gpt-4"),
    }
else:
    console.print("[bold red]Using HuggingFace API[/bold red]")
    llms = {
        "tiny": HfLLMInterface("gpt2"),
        "tiny": HfLLMInterface("gpt2"),
        "hq": HfLLMInterface("gpt2"),
    }
    print(llms["tiny"]([{"role": "user", "content": "Hello world!"}]))


class WorldObject(BaseModel):
    name: str
    metadata: dict | None


class World(BaseModel):
    objects: list[WorldObject]


class RenderObjectResponse(BaseModel):
    html: str


class RenderObjectRequest(BaseModel):
    world: World
    object: WorldObject


def fix_model_output(output, start="{", end="}"):
    # Find the first { and the last }
    first_bracket = output.find(start)
    last_bracket = output.rfind(end)
    result = output[first_bracket : last_bracket + 1]
    print(result)
    return result


class GenerateworldRequest(BaseModel):
    world_desc: str
    cond: World


@app.post("/gen_world")
def gen_world(request: GenerateworldRequest) -> World:
    world_desc = request.world_desc
    try:
        result = llms["regular"](
            [
                {
                    "role": "system",
                    "content": "You are WorldModelAI. Your purpose is to model"
                    " the videogamegame world as accurately as possible. The world is "
                    "represented as a JSON object that describes an "
                    "environment or scene. Objects do not have a position."
                    "There must be a game_state object that holds core variables"
                    "for the game loop and gameplay. The game_state object must also have"
                    "concrete variables that track the progress of the game and"
                    "conditions that the player will have to achieve to win"
                    " (these start as False) or conditions that will fail"
                    " the game if they are achieved (also start as False)."
                    " Do not use coordinates such as x and y, there are no"
                    " positions."
                    " The win conditions must match the objects in the world"
                    " for example, if there is a 'find the treasure' win"
                    " condition, there must be a treasure object in the world."
                    " Do not place a player object in the world, as this"
                    " will be handled by the game engine."
                    "Here is an example world:"
                    + json.dumps(
                        {
                            "objects": [
                                {
                                    "name": "tree",
                                    "metadata": {"color": "green"},
                                },
                                {
                                    "name": "game_state",
                                    "metadata": {
                                        "win_conditions": {"cut_tree": False},
                                        "fail_conditions": {"died_from_hunger": False},
                                    },
                                },
                            ]
                        }
                    )
                    + "\n The name of each object must be unique. The metadata"
                    " can be any JSON object. Only output the JSON object.",
                },
                {
                    "role": "user",
                    "content": "Generate the world" "for the description:" + world_desc,
                },
            ],
        )
        result = fix_model_output(result)

        world_dict = json.loads(result)
        world = World.parse_obj(world_dict)

        return world
    except Exception as e:
        return gen_world(world_desc)


@app.post("/render_object")
def render_object(request: RenderObjectRequest) -> RenderObjectResponse:
    try:
        world, object = request.world, request.object
        result = llms["regular"](
            [
                {
                    "role": "user",
                    "content": "World context: " + json.dumps(world.dict()),
                },
                {
                    "role": "system",
                    "content": "You are WebGameRendererAI. Your purpose is to render"
                    " world objects as HTML for a web game. These renders must be"
                    " simple but charming and must represent the state of the object."
                    " Note that there are some metadata fields that should not be"
                    " rendered. "
                    "For example, you must not show the age of an NPC"
                    " that the player has not met yet."
                    "The object must be rendered as an HTML div and can use Tailwind css classes. "
                    "They can use the style attribute for css and hardcoded svg or such."
                    "\n Feel free to make your renders detailed, colorful, and artistic."
                    " Only output the HTML. Do not use images or other external assets."
                    " You must not use position: absolute, position: fixed"
                    " css. This is because the game engine will position the objects"
                    " for you. Do not use width and height if your object contains"
                    " text because it might bleed. Note that we use a dark theme. "
                    "Only output the HTML.",
                },
                {
                    "role": "user",
                    "content": "Render this object now: " + json.dumps(object.dict()),
                },
            ],
        )
        result = fix_model_output(result, start="<", end=">")
        return RenderObjectResponse(html=result)
    except Exception as e:
        return render_object(request)


class ObjectTexturePromptGenerateRequest(BaseModel):
    world: World
    object: WorldObject


class ObjectTexturePromptGenerateResponse(BaseModel):
    prompt: str


@app.post("/object_texture_prompt")
def object_prompt(
    request: ObjectTexturePromptGenerateRequest,
) -> ObjectTexturePromptGenerateResponse:
    try:
        world, object = request.world, request.object
        console.print(object)
        result = llms["regular"](
            [
                {
                    "role": "user",
                    "content": "World context: " + json.dumps(world.dict()),
                },
                {
                    "role": "system",
                    "content": "You are GameRendererAI. Your purpose is to render"
                    " world objects as prompts that will be fed to an image generation"
                    "AI model. The object is represented as a JSON that you will"
                    " describe artistically"
                    "These prompts must be short and"
                    " descriptive and must represent the state of the object."
                    " Note that there are some metadata fields that should not be"
                    " rendered. "
                    " Here are some example prompts: "
                    " Aggresive shark, trending on Artstation\n"
                    " Potion stand, videogame asset, 4k texture\n"
                    " Coral reef, pink, destroyed\n"
                    " Healing potion, high quality videogame icon\n"
                    "Only output the prompt.",
                },
                {
                    "role": "user",
                    "content": "Render this object now. Output only the prompt: "
                    + json.dumps(object.dict()),
                },
            ],
        )
        console.print(result)
        return ObjectTexturePromptGenerateResponse(prompt=result)
    except Exception as e:
        return render_object(request)


class RoomTextureGenerateRequest(BaseModel):
    world_desc: str


class RoomTextureGenerateResponse(BaseModel):
    floor_texture: str
    wall_texture: str


@app.post("/gen_room_textures")
def room_textures(
    request: RoomTextureGenerateRequest,
) -> RoomTextureGenerateResponse:
    try:
        desc = request.world_desc
        console.print(desc)
        result = llms["regular"](
            [
                {
                    "role": "system",
                    "content": "You are GameRendererAI. Your purpose is to describe"
                    " game textures as prompts that will be fed to an image generation"
                    "AI model. In our game, there are multiple thematic rooms,"
                    " and each room has a floor and walls. You will be given"
                    " a description of the room and you must output a prompt"
                    " that describes the wall and floor textures. "
                    "These prompts must be short and"
                    " descriptive and must represent expectations that fit the"
                    " theme of the room. Your descriptions should be similar"
                    " to how textures are often described in online asset stores."
                    " You will return ONLY a JSON object with the keys floor_texture"
                    " and wall_texture. Here are some examples: \n"
                    "Underwater: \n"
                    + json.dumps(
                        {
                            "floor_texture": "Sandy sea floor, small plants, rocks, 4k texture",
                            "wall_texture": "Underwater, fish, corals",
                        }
                    )
                    + "\n"
                    "Forest: \n"
                    + json.dumps(
                        {
                            "floor_texture": "Grass, dirt, leaves, videogame asset",
                            "wall_texture": "Tress, side view, forest",
                        }
                    )
                    + "\n"
                    "Candyland: \n"
                    + json.dumps(
                        {
                            "floor_texture": "Chocolate soil, candy crush",
                            "wall_texture": "Pink dunes, side view",
                        }
                    )
                    + "\n"
                    "Only output the JSON object as shown and nothing else.",
                },
                {
                    "role": "user",
                    "content": "Generate the prompts for this theme now: "
                    + json.dumps(desc),
                },
            ],
        )
        console.print(result)
        result = fix_model_output(result, start="{", end="}")
        result = json.loads(result)
        return RoomTextureGenerateResponse.parse_obj(result)
    except Exception as e:
        console.print(e)
        return room_textures(request)


class ObjectInteraction(BaseModel):
    name: str
    display_name: str
    arguments: list[str] | None


class ObtainObjectInteractionsRequest(BaseModel):
    world: World
    object: WorldObject


class ObtainObjectInteractionsResponse(BaseModel):
    interactions: list[ObjectInteraction]


@app.post("/interact")
def obtain_object_interactions(
    request: ObtainObjectInteractionsRequest,
) -> ObtainObjectInteractionsResponse:
    try:
        world, object = request.world, request.object
        result = llms["regular"](
            [
                {
                    "role": "system",
                    "content": "You are GameMasterAI. Your purpose is to generate"
                    " interactions between the player and objects in the world. "
                    "These interactions"
                    " must be simple but entertaining and must be accurrate to the"
                    " expectations of the player. The interactions must be"
                    " represented as a JSON list where each element is an object"
                    " with a name, a display_name, and arguments. The name is a unique"
                    " identifier for the interaction and the display_name is"
                    " what the player sees. Interactions also have arguments, "
                    " which are a list of questions that the player must answer to"
                    " complete the interaction."
                    "Here is an example interaction:"
                    '{"name": "eat", "display_name": "Eat", "arguments": '
                    '["What do you want to eat?"]}',
                },
                {
                    "role": "user",
                    "content": "World context: " + json.dumps(world.dict()),
                },
                {
                    "role": "user",
                    "content": "Return JSON interactions for object: "
                    + json.dumps(object.dict()),
                },
            ],
        )
        print(result)
        result = fix_model_output(result, start="[", end="]")

        # Turn the result into a ObtainObjectInteractionsResponse object
        interactions_dict = json.loads(result)
        if isinstance(interactions_dict, list):
            interactions_dict = {"interactions": interactions_dict}

        # Add "custom" interaction which is always available and allows the player
        # to type in a custom command
        interactions_dict["interactions"].append(
            {
                "name": "custom",
                "display_name": "Custom",
                "arguments": ["What do you want to do?"],
            }
        )

        interactions = ObtainObjectInteractionsResponse.parse_obj(interactions_dict)

        return interactions
    except Exception as e:
        return obtain_object_interactions(request)


class DoInteractRequest(BaseModel):
    world: World
    object: WorldObject
    interaction: ObjectInteraction


class DeleteObject(BaseModel):
    name: str


CreateObject = WorldObject

OverwriteMetadata = WorldObject


class DisplayMessage(BaseModel):
    message: str


class DoInteractResponse(BaseModel):
    delete_objects: list[DeleteObject] | None
    create_objects: list[CreateObject] | None
    overwrite_metadata: list[OverwriteMetadata] | None
    display_messages: list[DisplayMessage] | None


@app.post("/do_interaction")
def do_interact(request: DoInteractRequest) -> DoInteractResponse:
    try:
        world, object, interaction = request.world, request.object, request.interaction
        result = llms["hq"](
            [
                {
                    "role": "user",
                    "content": "World context: " + json.dumps(world.dict()),
                },
                {
                    "role": "system",
                    "content": "You are GameEngineAI. Your purpose is to execute"
                    " interactions between the player and objects in the world. "
                    "The world, object, and interaction are provided as JSON."
                    "You must return an object of effects that the interaction"
                    " had on the world. You can use the game_state object"
                    " metadata to set global variables that track the game"
                    " progress. "
                    "Only output the JSON. You can set metadata values to None"
                    " to delete them."
                    "Here is an example result that uses all"
                    " possible effects: \n"
                    + json.dumps(
                        {
                            "delete_objects": [
                                {"name": "stick"},
                                {"name": "rock"},
                            ],
                            "create_objects": [
                                {
                                    "name": "axe",
                                    "metadata": {"color": "brown"},
                                },
                            ],
                            "overwrite_metadata": [
                                {
                                    "name": "player",
                                    "metadata": {"has_crafted": True},
                                },
                            ],
                            "display_messages": [
                                {"message": "Congrats!"},
                            ],
                        }
                    )
                    + "\nYou should prioritize creating and deleting objects "
                    "as this is more fun for the player. You should also challenge"
                    " the player by having certain interactions fail. For example,"
                    " if the player tries to eat a rock, you should show a message"
                    " that says 'You can't eat a rock!'. Some interactions should"
                    " also fail randomly, to represent a sense of difficulty."
                    " for example a cauldron might explode even if the player"
                    " correctly follows the recipe for a potion."
                    "You must not abuse display_messages by using it to display"
                    "things that didn't happen. For example, you can't say 'The"
                    "monster died!' if you don't also use delete_objects to delete"
                    "the monster. You must add new metadata and new objects"
                    " as the player discovers new things. For example, if the player"
                    " asks for the name of an NPC, you should add a new metadata"
                    " field to the NPC object that stores the name. This is "
                    "important, as otherwise you won't be able to remember"
                    " the name of the NPC later. You must also make sure to clean"
                    " up objects that are no longer needed. For example, if this"
                    " is a game about fixing cars and a car has already been fixed,"
                    " you should delete the broken car object.",
                },
                {
                    "role": "user",
                    "content": "Object context: " + json.dumps(object.dict()),
                },
                {
                    "role": "user",
                    "content": "Please return JSON effects for this interaction: "
                    + json.dumps(interaction.dict()),
                },
            ],
        )
        result = fix_model_output(result)

        # Turn the result into a DoInteractResponse object
        effects_dict = json.loads(result)
        return DoInteractResponse.parse_obj(effects_dict)
    except Exception as e:
        return do_interact(request)


class GameTickRequest(BaseModel):
    world: World


@app.post("/game_tick")
def do_interact(request: GameTickRequest) -> DoInteractResponse:
    try:
        world = request.world
        result = llms["hq"](
            [
                {
                    "role": "user",
                    "content": "World context: " + json.dumps(world.dict()),
                },
                {
                    "role": "system",
                    "content": "You are GameEngineAI. Your purpose is to compute"
                    " a game world tick."
                    "The world is provided as JSON."
                    "You must return a JSON object of effects that the tick"
                    " had on the world. You can use the game_state object"
                    " metadata to update global variables that track the game"
                    " progress. "
                    "Only output the JSON. You can set metadata values to None"
                    " to delete them."
                    "Here is an example result that uses all"
                    " possible effects: \n"
                    + json.dumps(
                        {
                            "delete_objects": [
                                {"name": "seed"},
                            ],
                            "create_objects": [
                                {
                                    "name": "plant",
                                    "metadata": {"growth": "1"},
                                },
                            ],
                            "overwrite_metadata": [
                                {
                                    "name": "calendar",
                                    "metadata": {"month": "february"},
                                },
                            ],
                            "display_messages": [
                                {"message": "Your plant grew!"},
                            ],
                        }
                    )
                    + "\nDuring the game tick, the world gets updated without"
                    " any player input. You should update the world to reflect"
                    " the change of time. For example, you can make plants grow"
                    ", or make the player hungry, or make enemies attack the"
                    "player, make loyalties change, etc. You should also"
                    " challenge the player by having new challenges and enemies"
                    " appear. For example, you can show a new quest, or a new"
                    " enemy, or a new friendly NPC. Do not perform any actions"
                    " that require player input. For example, if there is a"
                    " tree that the player can cut down, you should not cut"
                    " down the tree during the game tick. Instead, you should"
                    " make the tree grow older or something like that."
                    "You must not abuse display_messages by using it to display"
                    "things that didn't happen. For example, you can't say 'The"
                    "monster attacks you!' if you don't also use overwrite_metadata "
                    "to change the player's health. You also can't say something"
                    " like 'Helena is on her room' if there is no metadata that"
                    "backs this up. You must also make sure to clean"
                    " up objects that are no longer needed. For example, if this"
                    " is a game about fixing cars and a car has already been fixed,"
                    " you should delete the broken car object.",
                },
                {
                    "role": "user",
                    "content": "Please return JSON effects for this game tick now:",
                },
            ],
        )
        result = fix_model_output(result)

        # Turn the result into a DoInteractResponse object
        effects_dict = json.loads(result)
        return DoInteractResponse.parse_obj(effects_dict)
    except Exception as e:
        return do_interact(request)


# == 3D generation ==
class MeshInferenceRequest(BaseModel):
    text: str


class MeshInferenceResponse(BaseModel):
    obj: str


xm = load_model("transmitter", device="cuda")
shap_e = load_model("text300M", device="cuda")
shap_e_diffusion = diffusion_from_config(load_config("diffusion"))


@app.post("/generate_mesh_shap_e")
@torch.no_grad()
def generate_shap_e(request: MeshInferenceRequest) -> MeshInferenceResponse:
    console.print(request)
    t = time.perf_counter()
    batch_size = 1
    guidance_scale = 15.0

    latents = sample_latents(
        batch_size=batch_size,
        model=shap_e,
        diffusion=shap_e_diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[request.text] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    # Example of saving the latents as meshes.
    f = StringIO()
    tm = decode_latent_mesh(xm, latents[0]).tri_mesh()
    tm.verts[..., 1] -= tm.verts[..., 1].min()
    tm.write_obj(f)
    obj = f.getvalue()
    console.print(f"Generated mesh in", time.perf_counter() - t, "seconds")
    return MeshInferenceResponse(obj=obj)
