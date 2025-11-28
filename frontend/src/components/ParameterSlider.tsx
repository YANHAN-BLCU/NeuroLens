type Props = {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
  onChange: (value: number) => void;
  description?: string;
};

export function ParameterSlider({
  label,
  value,
  min,
  max,
  step = 0.01,
  suffix,
  onChange,
  description,
}: Props) {
  const showInteger =
    Number.isInteger(step) &&
    Number.isInteger(min) &&
    Number.isInteger(max);
  const displayValue = showInteger
    ? Math.round(value).toString()
    : value.toFixed(2);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm text-slate-300">
        <span className="font-medium">{label}</span>
        <span className="tabular-nums text-accent-soft">
          {displayValue}
          {suffix}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full accent-accent"
      />
      {description ? (
        <p className="text-xs text-slate-500">{description}</p>
      ) : null}
    </div>
  );
}

