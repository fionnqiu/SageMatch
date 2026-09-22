import { useEffect, useState } from "react";
import { Moon, Settings, Sun } from "lucide-react";
import {
  MorphPopover,
  MorphPopoverContent,
  MorphPopoverTrigger,
} from "../components/motion/popover-morph";
import { applyTheme, readTheme, type ThemeMode } from "../lib/theme";

const THEMES: { mode: ThemeMode; label: string; icon: typeof Moon }[] = [
  { mode: "dark", label: "深色", icon: Moon },
  { mode: "light", label: "浅色", icon: Sun },
];

/** 管理端和用户端共用的外观菜单。选择写在 html 上，两边看到的是同一套主题。 */
export function ThemeMenu() {
  const [theme, setTheme] = useState<ThemeMode>(readTheme);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  return (
    <MorphPopover>
      <MorphPopoverTrigger>
        <button
          type="button"
          className="flex h-7 w-7 items-center justify-center rounded-md text-dim hover:bg-field hover:text-ink"
          aria-label="设置"
        >
          <Settings size={14} />
        </button>
      </MorphPopoverTrigger>
      <MorphPopoverContent side="top" align="end" className="w-44 p-2">
        <div className="px-2 pb-1.5 text-[10px] font-medium text-dim">外观</div>
        <div className="flex gap-1">
          {THEMES.map((item) => {
            const Icon = item.icon;
            const on = theme === item.mode;
            return (
              <button
                key={item.mode}
                type="button"
                onClick={() => setTheme(item.mode)}
                aria-pressed={on}
                className={`flex h-8 flex-1 items-center justify-center gap-1.5 rounded-md text-[11px] ${
                  on ? "bg-forest font-semibold text-mint-3" : "text-mute hover:bg-muted"
                }`}
              >
                <Icon size={13} />
                {item.label}
              </button>
            );
          })}
        </div>
      </MorphPopoverContent>
    </MorphPopover>
  );
}
