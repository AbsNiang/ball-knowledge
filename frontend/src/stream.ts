import type { Event } from '../../contracts/generated';
import { reduceEvent, type Session } from './state';

export type ConnectionState = 'connecting' | 'connected' | 'reconnecting' | 'failed';
interface Callbacks {
  session: (session: Session) => void;
  connection: (state: ConnectionState) => void;
  missing: () => void;
  error: (message: string) => void;
  command: (message: {type: string; request_id: string; detail?: {message?: string}}) => void;
}

/** One stream per run. Reconnect from the last applied event, never resend commands. */
export class RunStream {
  private socket?: WebSocket;
  private timer?: ReturnType<typeof setTimeout>;
  private watchdog?: ReturnType<typeof setTimeout>;
  private heartbeat?: ReturnType<typeof setTimeout>;
  private stopped = false;
  private attempts = 0;

  constructor(private state: Session, private callbacks: Callbacks) {
    this.connect();
  }

  private connect() {
    this.callbacks.connection(this.attempts ? 'reconnecting' : 'connecting');
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${location.host}/api/runs/${encodeURIComponent(this.state.id)}/events?from_seq=${this.state.seq + 1}`);
    this.socket = socket;
    const reconnect = () => {
      if (this.stopped || this.socket !== socket) return;
      clearTimeout(this.watchdog);
      clearTimeout(this.heartbeat);
      this.socket = undefined;
      socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null;
      socket.close();
      if (this.attempts >= 5) {
        this.callbacks.connection('failed');
        return;
      }
      this.callbacks.connection('reconnecting');
      this.timer = setTimeout(() => this.connect(), Math.min(500 * 2 ** this.attempts++, 8000));
    };
    this.watchdog = setTimeout(reconnect, 8000);
    const ping = () => {
      if (this.stopped || this.socket !== socket || socket.readyState !== WebSocket.OPEN) return;
      socket.send(JSON.stringify({type:'ping'}));
      this.watchdog = setTimeout(reconnect, 5000);
    };
    socket.onopen = () => {
      if (this.stopped || this.socket !== socket) return;
      clearTimeout(this.watchdog);
      this.heartbeat = setTimeout(ping, 15000);
      this.callbacks.connection('connected');
    };
    socket.onmessage = ({data}) => {
      if (this.stopped || this.socket !== socket) return;
      try {
        const message = JSON.parse(data);
        if (message?.type === 'pong') {
          clearTimeout(this.watchdog);
          this.attempts = 0;
          this.heartbeat = setTimeout(ping, 15000);
          return;
        }
        if (message?.type === 'command_ack' || message?.type === 'command_error') {
          this.callbacks.command(message);
          return;
        }
        if (message?.detail?.code === 'run_not_found') {
          this.close();
          this.callbacks.missing();
          return;
        }
        if (message?.detail) throw new Error(message.detail.message || 'Stream rejected.');
        if (!Number.isInteger(message?.seq) || message.seq < 1 || message.run_id !== this.state.id ||
            !message.payload || message.payload.type !== message.type) {
          throw new Error('Invalid event received from the backend.');
        }
        this.state = reduceEvent(this.state, message as Event);
        this.attempts = 0;
        this.callbacks.session(this.state);
      } catch (error) {
        this.callbacks.error(error instanceof Error ? error.message : 'Invalid stream message.');
        this.close();
        this.callbacks.connection('failed');
      }
    };
    // Browsers follow a socket error with close; schedule only once, on close.
    socket.onerror = () => {};
    socket.onclose = reconnect;
  }

  send(message: unknown): boolean {
    if (this.stopped || this.socket?.readyState !== WebSocket.OPEN) return false;
    this.socket.send(JSON.stringify(message));
    return true;
  }

  retry() {
    this.close();
    this.stopped = false;
    this.attempts = 0;
    this.connect();
  }

  close() {
    this.stopped = true;
    clearTimeout(this.timer);
    clearTimeout(this.watchdog);
    clearTimeout(this.heartbeat);
    const socket = this.socket;
    this.socket = undefined;
    if (!socket) return;
    socket.onmessage = socket.onclose = socket.onerror = null;
    // React StrictMode may clean up before the handshake finishes. Let it finish
    // before closing to avoid aborting Vite's in-flight proxy upgrade.
    if (socket.readyState === WebSocket.CONNECTING) socket.onopen = () => socket.close();
    else { socket.onopen = null; socket.close(); }
  }
}
