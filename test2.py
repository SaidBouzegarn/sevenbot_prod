# Cell 1: Imports and setup
from pydantic import BaseModel, Field, create_model
from typing import List, Optional, Any, Literal, Annotated
from langgraph.graph import add_messages

# Cell 2: Define sample agent classes
class Level1Agent:
    def __init__(self, name):
        self.name = name
        self.state_schema = self._create_dynamic_state_schema()

    def _create_dynamic_state_schema(self):
        return create_model(
            f'{self.name}_Level1State',
            **{
                "level1_2_conversation": (Annotated[List, add_messages], ...),
                "level1_3_conversation": (Annotated[List, add_messages], ...),
                f"{self.name}_is_first_execution": (bool, Field(default=True)),
                f"{self.name}_assistant_conversation": (Annotated[List, add_messages], ...),
                f"{self.name}_domain_knowledge": (Optional[Any], None),
                f"{self.name}_mode": (Literal["research", "converse"], Field(default="research")),
                f"{self.name}_messages": (Annotated[List, add_messages], ...),
                f"{self.name}_runs_counter": (int, Field(default=0)),
                f"{self.name}_assistant_runs_counter": (int, Field(default=0)),
            },
            __base__=BaseModel
        )

class Level2Agent:
    def __init__(self, name):
        self.name = name
        self.state_schema = self._create_dynamic_state_schema()

    def _create_dynamic_state_schema(self):
        return create_model(
            f'{self.name}_Level2State',
            **{
                "level1_2_conversation": (Annotated[List, add_messages], ...),
                "level2_3_conversation": (Annotated[List, add_messages], ...),
                f"{self.name}_is_first_execution": (bool, Field(default=True)),
                f"{self.name}_messages": (Annotated[List, add_messages], ...),
                f"{self.name}_mode": (Literal["aggregate_for_ceo", "break_down_for_executives"], Field(default="break_down_for_executives")),
                f"{self.name}_runs_counter": (int, Field(default=0)),
            },
            __base__=BaseModel
        )

class Level3Agent:
    def __init__(self, name):
        self.name = name
        self.state_schema = self._create_dynamic_state_schema()

    def _create_dynamic_state_schema(self):
        return create_model(
            'Level3State',
            level2_3_conversation=(Annotated[List, add_messages], ...),
            level1_3_conversation=(Annotated[List, add_messages], ...),
            company_knowledge=(str, ...),
            news_insights=(List[str], ...),
            digest=(List[str], ...),
            action=(List[str], ...),
            ceo_messages=(Annotated[List, add_messages], ...),
            ceo_assistant_conversation=(Annotated[List, add_messages], ...),
            ceo_mode=(Literal["research_information", "write_to_digest", "communicate_with_directors", "communicate_with_executives", "end"], ...),
            level2_agents=(List[str], ...),
            level1_agents=(List[str], ...),
            is_first_execution=(bool, Field(default=True)),
            ceo_runs_counter=(int, Field(default=0)),
            runs_counter=(int, Field(default=0)),
            ceo_assistant_runs_counter=(int, Field(default=0)),
            __base__=BaseModel
        )

# Cell 3: Create sample agents
level1_agents = [Level1Agent(f"agent{i}") for i in range(1, 5)]
level2_agents = [Level2Agent(f"supervisor{i}") for i in range(1, 3)]
ceo_agent = Level3Agent("ceo")

# Cell 4: Define create_unified_state_schema function
def create_unified_state_schema(level1_agents, level2_agents, ceo_agent):
    unified_fields = {}

    def add_agent_fields(agent):
        for field_name, field in agent.state_schema.model_fields.items():
            if field_name not in unified_fields:
                # Make all fields optional with a default value
                if field.annotation == List or field.annotation == Annotated[List, add_messages]:
                    unified_fields[field_name] = (Optional[field.annotation], Field(default_factory=list))
                elif field.annotation == str:
                    unified_fields[field_name] = (Optional[str], Field(default=""))
                elif field.annotation == bool:
                    unified_fields[field_name] = (bool, Field(default=field.default))
                elif field.annotation == int:
                    unified_fields[field_name] = (int, Field(default=field.default))
                elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == Literal:
                    unified_fields[field_name] = (Optional[field.annotation], Field(default=None))
                else:
                    unified_fields[field_name] = (Optional[field.annotation], Field(default=None))

    # Add fields from all agents
    for agent in level1_agents + level2_agents + [ceo_agent]:
        add_agent_fields(agent)

    # Create and return the unified state schema
    UnifiedState = create_model("UnifiedState", **unified_fields, __base__=BaseModel)
    return UnifiedState

# Cell 5: Create unified state schema
UnifiedState = create_unified_state_schema(level1_agents, level2_agents, ceo_agent)

# Cell 6: Print unified state schema fields
print("Unified State Schema Fields:")
for field_name, field in UnifiedState.model_fields.items():
    print(f"{field_name}: {field.annotation} (default: {field.default})")

# Cell 7: Create initial state
initial_state = UnifiedState()

# Cell 8: Print initial state
print("\nInitial State:")
for field_name, value in initial_state:
    print(f"{field_name}: {value}")

# Cell 9: Set some values and print again
initial_state.company_knowledge = "Our company is a leading tech firm."
initial_state.news_insights = ["Recent AI advancements"]
initial_state.ceo_mode = "research_information"
initial_state.level2_agents = ["supervisor1", "supervisor2"]
initial_state.level1_agents = ["agent1", "agent2", "agent3", "agent4"]

print("\nUpdated Initial State:")
for field_name, value in initial_state:
    print(f"{field_name}: {value}")
