'use client';

import React from 'react';
import {
  AlertCircleIcon,
  PaperclipIcon,
  UploadIcon,
  XIcon,
  ShieldAlertIcon,
  ShieldCheckIcon,
  DownloadIcon,
} from 'lucide-react';

import { Button, Badge, Label } from '@/components/ui';
import { Textarea } from '@/components/ui/textarea';
import { formatBytes, useFileUpload } from '@/hooks/use-file-upload';
import api from '@/lib/api';

type RowOut = {
  // original columns (unknown shape)
  [k: string]: any;
  Amount_Paid_Log?: number;
  Payment_Currency_Unique?: number;
  'Laundering Probability': number;
  'Predicted Label': number;
  Explanation: string;
};

const fmtPct = (p: number) => `${(p * 100).toFixed(2)}%`;

function RiskBadge({ label }: { label: number }) {
  const laundering = label === 1;
  return laundering ? (
    <Badge variant="default" className="gap-1.5 bg-rose-600">
      <ShieldAlertIcon className="size-3 text-black" />
      <span className="font-semibold text-black">Suspicious</span>
    </Badge>
  ) : (
    <Badge variant="default" className="gap-1.5 bg-emerald-500">
      <ShieldCheckIcon className="size-3 text-black" />
      <span className="font-semibold text-black">Legitimate</span>
    </Badge>
  );
}

function downloadCSV(filename: string, rows: RowOut[]) {
  if (!rows?.length) return;
  const headers = Array.from(
    rows.reduce<Set<string>>((acc, r) => {
      Object.keys(r).forEach((k) => acc.add(k));
      return acc;
    }, new Set<string>())
  );

  const escape = (v: any) => {
    if (v === null || v === undefined) return '';
    const s = String(v);
    if (s.includes('"') || s.includes(',') || s.includes('\n')) {
      return `"${s.replace(/"/g, '""')}"`;
    }
    return s;
  };

  const csv = [
    headers.join(','),
    ...rows.map((r) => headers.map((h) => escape(r[h])).join(',')),
  ].join('\n');

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

const Arrest = () => {
  const maxSize = 100 * 1024 * 1024; // 10MB
  const [
    { files, isDragging, errors },
    {
      handleDragEnter,
      handleDragLeave,
      handleDragOver,
      handleDrop,
      openFileDialog,
      removeFile,
      getInputProps,
    },
  ] = useFileUpload({
    maxSize,
    multiple: false,
    accept: '.csv,text/csv',
  });

  const file = files[0];

  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [rows, setRows] = React.useState<RowOut[] | null>(null);

  function clearSelection() {
    if (file) removeFile(file.id);
    setRows(null);
    setError(null);
  }

  async function runScoring(asCsv = false) {
    if (!file || !(file.file instanceof File)) {
      setError('Please choose a CSV file.');
      return;
    }
    setError(null);
    setLoading(true);
    setRows(null);

    try {
      const formData = new FormData();
      formData.append('file', file.file);

      if (asCsv) {
        // server-side CSV
        const res = await api.post(`/predict-csv?as_csv=true`, formData, {
          responseType: 'blob',
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        const dlUrl = URL.createObjectURL(res.data);
        const a = document.createElement('a');
        a.href = dlUrl;
        a.download = `aml_results_${file.file.name}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(dlUrl);
      } else {
        // JSON for in-app render
        const res = await api.post<RowOut[]>(`/predict-csv`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        setRows(res.data || []);
      }
    } catch (e: any) {
      const detail =
        e?.response?.data?.detail || e?.message || 'CSV prediction failed';
      setError(detail);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col items-center mx-4 my-20">
      <div className="py-8 space-y-6 w-full max-w-4xl">
        <h1 className="text-2xl font-semibold">Detect Money Laundering</h1>
        <p className="text-sm text-muted-foreground -mt-3">
          Upload a CSV with columns like <code>Amount Paid</code>,{' '}
          <code>Payment Currency</code>, and more.
        </p>

        {/* Drop area */}
        <div
          role="button"
          onClick={openFileDialog}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          data-dragging={isDragging || undefined}
          className="border-input hover:bg-accent-fog data-[dragging=true]:bg-accent/50 has-[input:focus]:border-ring has-[input:focus]:ring-ring/50 flex min-h-40 flex-col items-center justify-center rounded-md border border-dashed p-4 transition-colors has-disabled:pointer-events-none has-disabled:opacity-50 has-[input:focus]:ring-[3px]"
          title="Drop a CSV file or click to browse"
        >
          <input
            {...getInputProps({
              accept: '.csv,text/csv',
              multiple: false,
            })}
            className="sr-only"
            aria-label="Upload CSV file"
            disabled={Boolean(file)}
          />

          <div className="flex flex-col items-center justify-center text-center">
            <div
              className="bg-background mb-2 flex size-11 shrink-0 items-center justify-center rounded-full border"
              aria-hidden="true"
            >
              <UploadIcon className="size-4 opacity-60" />
            </div>
            <p className="mb-1.5 text-sm font-medium">Upload CSV</p>
            <p className="text-muted-foreground text-base">
              Drag & drop or click to browse (max. {formatBytes(maxSize)})
            </p>
          </div>
        </div>

        {/* Errors */}
        {(errors.length > 0 || error) && (
          <div
            className="text-destructive flex items-center gap-1 text-sm"
            role="alert"
          >
            <AlertCircleIcon className="size-4" />
            <span>{error ?? errors[0]}</span>
          </div>
        )}

        {/* File chip + actions */}
        {file && (
          <div className="flex items-center justify-between gap-2 rounded-md border border-accent-nvidia border-dashed px-4 py-2">
            <div className="flex items-center gap-3 overflow-hidden">
              <PaperclipIcon
                className="size-4 shrink-0 opacity-60"
                aria-hidden="true"
              />
              <div className="min-w-0">
                <p className="truncate text-[13px] font-medium">
                  {file.file.name}
                </p>
                <p className="text-base text-muted-foreground">
                  {(file.file as File).type || 'text/csv'} •{' '}
                  {formatBytes((file.file as File).size)}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => runScoring(false)}
                disabled={loading}
                className="rounded-md px-4 py-2 bg-accent-nvidia text-sm font-bold text-black/80 border border-white/5 cursor-pointer hover:bg-accent-nvidia-dim transition-all active:scale-[99%]"
              >
                {loading ? 'Exposing Fraud..' : 'Expose Fraud'}
              </Button>

              <Button
                size="sm"
                variant="outline"
                onClick={() => runScoring(true)}
                disabled={loading}
                className="rounded-md px-4 py-2 bg-accent-fog text-sm font-bold text-white border border-white/5 cursor-pointer hover:bg-accent-mist transition-all active:scale-[99%]"
                title="Download a CSV produced by the server"
              >
                <DownloadIcon className="size-4 mr-1.5" />
                Download CSV (server)
              </Button>

              <Button
                size="icon"
                variant="ghost"
                className="text-muted-foreground/80 hover:text-foreground -me-2 size-8 hover:bg-transparent"
                onClick={clearSelection}
                aria-label="Remove file"
              >
                <XIcon className="size-4" aria-hidden="true" />
              </Button>
            </div>
          </div>
        )}

        {/* Results */}
        {rows && rows.length > 0 && (
          <div className="mt-6">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-lg font-medium">Results ({rows.length})</h2>
              <Button
                size="sm"
                variant="outline"
                onClick={() =>
                  downloadCSV(
                    `aml_results_${file?.file.name || 'results'}`,
                    rows
                  )
                }
                className="rounded-md px-4 py-2 bg-accent-fog text-sm font-bold text-white border border-white/5 cursor-pointer hover:bg-accent-mist transition-all active:scale-[99%]"
                title="Download a CSV created on the client"
              >
                <DownloadIcon className="size-4 mr-1.5" />
                Download CSV (client)
              </Button>
            </div>

            <div className="grid gap-3">
              {rows.map((r, idx) => {
                const prob = Number(r['Laundering Probability'] ?? 0);
                const label = Number(r['Predicted Label'] ?? 0);
                const explanation = String(r['Explanation'] ?? '');
                const amt =
                  r['Amount Paid'] ?? r['Amount_Paid'] ?? r['Amount'] ?? '—';
                const cur =
                  r['Payment Currency'] ??
                  r['Payment_Currency'] ??
                  r['Currency'] ??
                  '—';
                const payFmt =
                  r['Payment Format'] ?? r['Payment_Format'] ?? '—';
                const ts = r['Timestamp'] ?? r['Date'] ?? '—';

                return (
                  <div
                    key={idx}
                    className={`bg-gradient-to-bl ${
                      label === 1 ? 'from-rose-700' : 'from-emerald-700'
                    } from-0% to-40% to-accent-charcol rounded-xl`}
                  >
                    <div className="rounded-lg border border-accent-fog p-4 space-y-3 backdrop-blur-3xl">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <RiskBadge label={label} />
                          <span className="text-sm text-muted-foreground">
                            Row #{idx + 1}
                          </span>
                        </div>
                        <div className="text-right">
                          <div className="text-base">
                            Laundering probability
                          </div>
                          <div className="text-base font-semibold text-accent-nvidia">
                            {fmtPct(prob)}
                          </div>
                        </div>
                      </div>
                      <div className="text-base text-muted-foreground">
                        <span className="mr-3">
                          Timestamp:{' '}
                          <b className="text-foreground">{String(ts)}</b>
                        </span>
                        <span className="mr-3">
                          Amount Paid:{' '}
                          <b className="text-foreground">{String(amt)}</b>
                        </span>
                        <span className="mr-3">
                          Payment Currency:{' '}
                          <b className="text-foreground">{String(cur)}</b>
                        </span>
                        <span className="mr-3">
                          Payment Format:{' '}
                          <b className="text-foreground">{String(payFmt)}</b>
                        </span>
                      </div>
                      <div className="h-2 w-full rounded bg-white/10 overflow-hidden">
                        <div
                          className={`h-2 ${
                            label === 1 ? 'bg-rose-500' : 'bg-emerald-500'
                          }`}
                          style={{ width: `${Math.min(100, prob * 100)}%` }}
                        />
                      </div>
                      <div>
                        <Label className="text-sm mb-1 inline-block">
                          Explanation
                        </Label>
                        <Textarea
                          readOnly
                          className="min-h-40 bg-accent-dusk/30 border-accent-fog text-sm scroll"
                          value={explanation}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Arrest;
