import { useState, type ReactNode } from "react";
import Sidebar, { type PageId } from "@/components/Sidebar";
import PulsePage from "@/pages/PulsePage";
import SkillsPage from "@/pages/SkillsPage";
import AgentsPage from "@/pages/AgentsPage";
import DataPage from "@/pages/DataPage";
import TargetsPage from "@/pages/TargetsPage";
import SettingsPage from "@/pages/SettingsPage";

const pages: Record<PageId, () => ReactNode> = {
  pulse: PulsePage,
  skills: SkillsPage,
  agents: AgentsPage,
  data: DataPage,
  targets: TargetsPage,
  settings: SettingsPage,
};

export default function App() {
  const [page, setPage] = useState<PageId>("pulse");
  const Page = pages[page];

  return (
    <div className="flex min-h-screen bg-zinc-925">
      <Sidebar active={page} onNav={setPage} />
      <main className="flex-1 p-8 overflow-auto bg-zinc-900">
        <div className="max-w-5xl">
          <Page />
        </div>
      </main>
    </div>
  );
}
