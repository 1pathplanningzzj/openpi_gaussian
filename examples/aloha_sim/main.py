# import dataclasses
# import logging
# import pathlib

# import env as _env
# from openpi_client import action_chunk_broker
# from openpi_client import websocket_client_policy as _websocket_client_policy
# from openpi_client.runtime import runtime as _runtime
# from openpi_client.runtime.agents import policy_agent as _policy_agent
# import saver as _saver
# import tyro


# @dataclasses.dataclass
# class Args:
#     out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/videos")

#     task: str = "gym_aloha/AlohaTransferCube-v0"
#     seed: int = 0

#     action_horizon: int = 10

#     host: str = "0.0.0.0"
#     port: int = 8000

#     display: bool = False


# def main(args: Args) -> None:
#     runtime = _runtime.Runtime(
#         environment=_env.AlohaSimEnvironment(
#             task=args.task,
#             seed=args.seed,
#         ),
#         agent=_policy_agent.PolicyAgent(
#             policy=action_chunk_broker.ActionChunkBroker(
#                 policy=_websocket_client_policy.WebsocketClientPolicy(
#                     host=args.host,
#                     port=args.port,
#                 ),
#                 action_horizon=args.action_horizon,
#             )
#         ),
#         subscribers=[
#             _saver.VideoSaver(args.out_dir),
#         ],
#         max_hz=50,
#     )

#     runtime.run()


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, force=True)
#     tyro.cli(main)
import dataclasses
import logging
import pathlib
import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
import tyro

@dataclasses.dataclass
class Args:
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/videos")
    # task: str = "gym_aloha/AlohaTransferCube-v0"
    task: str = "gym_aloha/AlohaInsertion-v0"

    
    seed: int = 0
    action_horizon: int = 10
    host: str = "0.0.0.0"
    port: int = 8000
    display: bool = False
    num_episodes: int = 50 

def main(args: Args) -> None:
    # 【新增循环】开始循环跑 num_episodes 次
    for i in range(args.num_episodes):
        logging.info(f"=== Starting Run {i+1}/{args.num_episodes} ===")
        
        # 注意：这里 seed = args.seed + i
        # 这样每次环境初始化的物体位置才不一样
        runtime = _runtime.Runtime(
            environment=_env.AlohaSimEnvironment(
                task=args.task,
                seed=args.seed + i, 
            ),
            agent=_policy_agent.PolicyAgent(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=_websocket_client_policy.WebsocketClientPolicy(
                        host=args.host,
                        port=args.port,
                    ),
                    action_horizon=args.action_horizon,
                )
            ),
            subscribers=[
                _saver.VideoSaver(args.out_dir),
            ],
            max_hz=50,
        )
        runtime.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
