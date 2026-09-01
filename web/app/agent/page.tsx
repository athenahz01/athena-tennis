import AgentDesk, { type AgentState } from "@/components/AgentDesk";
import agent from "@/public/data/agent.json";

export default function AgentPage() {
  return <AgentDesk fallback={agent as AgentState} />;
}
