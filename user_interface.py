import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter.filedialog import askdirectory
import epilepsy_recognition
import os
import time
import threading


class Console(tk.Frame):
    """Simple console that can execute bash commands"""

    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)

        self.text_options = {"state": "disabled",
                             "bg": "white",
                             "fg": "black",
                             "insertbackground": "#08c614",
                             "selectbackground": "#f01c1c"}

        self.text = ScrolledText(self, **self.text_options)

        # It seems not to work when Text is disabled...
        # self.text.bind("<<Modified>>", lambda: self.text.frame.see(tk.END))

        self.text.pack(expand=True, fill="both")

        # bash command, for example 'ping localhost' or 'pwd'
        # that will be executed when "Execute" is pressed
        self.command = ""
        self.popen = None     # will hold a reference to a Popen object
        self.running = False  # True if the process is running

        self.bottom = tk.Frame(self)

        # self.prompt = tk.Label(self.bottom, text="Enter the command: ")
        # self.prompt.pack(side="left", fill="x")
        # self.entry = tk.Entry(self.bottom)
        # self.entry.bind("<Return>", self.start_thread)
        # self.entry.bind("<Command-a>", lambda e: self.entry.select_range(0, "end"))
        # self.entry.bind("<Command-c>", self.clear)
        # self.entry.focus()
        # self.entry.pack(side="left", fill="x", expand=True)

        # self.executer = tk.Button(self.bottom, text="Execute", command=self.start_thread)
        # self.executer.pack(side="left", padx=5, pady=2)
        # self.clearer = tk.Button(self.bottom, text="Clear", command=self.clear)
        # self.clearer.pack(side="left", padx=5, pady=2)
        # self.stopper = tk.Button(self.bottom, text="Stop", command=self.stop)
        # self.stopper.pack(side="left", padx=5, pady=2)

        self.bottom.pack(side="bottom", fill="both")

    def clear_text(self):
        """Clears the Text widget"""
        self.text.config(state="normal")
        self.text.delete(1.0, "end-1c")
        self.text.config(state="disabled")

    def clear(self, event=None):
        """Does not stop an eventual running process,
        but just clears the Text and Entry widgets."""
        self.clear_text()

    def show(self, message):
        """Inserts message into the Text wiget"""
        self.text.config(state="normal")
        self.text.insert("end", message)
        self.text.see("end")
        self.text.config(state="disabled")

    def start_thread(self):
        """Starts a new thread and calls process"""
        self.stop()
        self.running = True
        # self.process is called by the Thread's run method
        threading.Thread(target=self.process).start()

    def process(self):
        """Runs in an infinite loop until self.running is False"""
        while self.running:
            self.execute()

    def stop(self):
        """Stops an eventual running process"""
        if self.popen:
            try:
                self.popen.kill()
            except ProcessLookupError:
                pass
        self.running = False

    def execute(self, command):
        """Keeps inserting line by line into self.text
        the output of the execution of self.command"""
        try:
            # self.popen is a Popen object
            self.popen = Popen(command.split(), stdout=PIPE, bufsize=1)
            lines_iterator = iter(self.popen.stdout.readline, b"")

            # poll() return None if the process has not terminated
            # otherwise poll() returns the process's exit code
            while self.popen.poll() is None:
                for line in lines_iterator:
                    self.show(line.decode("utf-8"))
            self.show("Process " + command + " terminated.\n\n")

        except FileNotFoundError:
            self.show("Unknown command: " + command + "\n\n")
        except IndexError:
            self.show("No command entered\n\n")

        self.stop()

def formatForm(form, width, heigth):
    """设置居中显示"""
    # 得到屏幕宽度
    win_width = form.winfo_screenwidth()
    # 得到屏幕高度
    win_higth = form.winfo_screenheight()

    # 计算偏移量
    width_adjust = (win_width - width) / 2
    higth_adjust = (win_higth - heigth) / 2

    form.geometry("%dx%d+%d+%d" % (width, heigth, width_adjust, higth_adjust))


class LoadingBar(object):

    def __init__(self, width=200):
        # 存储显示窗体
        self.__dialog = None
        # 记录显示标识
        self.__showFlag = True
        # 设置滚动条的宽度
        self.__width = width
        # 设置窗体高度
        self.__heigth = 20

    def show(self, speed=10, sleep=0):
        """显示的时候支持重置滚动条速度和标识判断等待时长"""
        # 防止重复创建多个
        if self.__dialog is not None:
            return

        # 线程内读取标记的等待时长（单位秒）
        self.__sleep = sleep

        # 创建窗体
        self.__dialog = tk.Toplevel()
        # 去除边框
        self.__dialog.overrideredirect(-1)
        # 设置置顶
        self.__dialog.wm_attributes("-topmost", True)
        formatForm(self.__dialog, self.__width, self.__heigth)
        # 实际的滚动条控件
        self.bar = ttk.Progressbar(self.__dialog, length=self.__width, mode="indeterminate",
                                   orient=tk.HORIZONTAL)
        self.bar.pack(expand=True)
        # 数值越小，滚动越快
        self.bar.start(speed)
        # 开启新线程保持滚动条显示
        t = threading.Thread(target=self.waitClose)
        t.setDaemon(True)
        t.start()

    def waitClose(self):
        # 控制在线程内等待回调销毁窗体
        while self.__showFlag:
            time.sleep(self.__sleep)

        # 非空情况下销毁
        if self.__dialog is not None:
            self.__dialog.destroy()

        # 重置必要参数
        self.__dialog = None
        self.__showFlag = True

    def close(self):
        # 设置显示标识为不显示
        self.__showFlag = False


def set_default(source_dir_text, output_dir_text, expert_entry, subject_entry, preprocess_cb, predict_cb, console):
    source_dir_text["text"] = ""
    output_dir_text["text"] = ""
    expert_entry.delete(0, tk.END)
    subject_entry.delete(0, tk.END)
    # extract_move_cb.select()
    preprocess_cb.select()
    predict_cb.select()
    console.clear()


def select_directory(txt_label):
    file_path = askdirectory()
    if not file_path:
        return
    txt_label["text"] = file_path


def run(source_dir_text, output_dir_text, expert_entry, subject_entry, preprocess_var, predict_var, console):
    source_dir = source_dir_text["text"]
    output_dir = output_dir_text["text"]
    expert = expert_entry.get().strip()
    subject = subject_entry.get().strip()
    # extract_move = True if extract_move_var.get() == 1 else 0
    preprocess_var = True if preprocess_var.get() == 1 else 0
    predict_var = True if predict_var.get() == 1 else 0
    if not source_dir or not output_dir or not expert or not subject:
        messagebox.showerror(title="Information Error", message="You have to fill all needed information.")
        return False
    # print("source_dir: {}\noutput_dir: {}\nexpert: {}\nsubject: {}\npreprocess_var: {}\npredict_var: {}\n".format(
    #     source_dir, output_dir, expert, subject, preprocess_var, predict_var))
    # epilepsy_recognition.main(source_dir, output_dir, expert, subject, extract_move, preprocess_var, predict_var)
    pj_dir = os.path.dirname(os.path.realpath(__file__))
    main_path = os.path.join(pj_dir, "epilepsy_recognition.py")
    command = "python {} --source-dir {} --output-dir {} --expert {} --subject {}".format(
        main_path, source_dir, output_dir, expert, subject)
    if preprocess_var:
        command += " --preprocess"
    if predict_var:
        command += " --predict"
    console.execute(command)
    return True


def main():
    window = tk.Tk()
    window.title("Epilepsy Recognition")
    information_frame = tk.Frame(window)

    source_dir_label = tk.Label(master=information_frame, text="Source Directory: ")
    source_dir_label.grid(row=0, column=0)
    source_dir_text = tk.Label(master=information_frame, text="")
    source_dir_text.grid(row=0, column=1)
    source_dir_button = tk.Button(master=information_frame, text="Select",
                                  command=lambda: select_directory(source_dir_text))
    source_dir_button.grid(row=0, column=2)
    information_frame.pack()

    output_dir_label = tk.Label(master=information_frame, text="Output Directory: ")
    output_dir_label.grid(row=1, column=0)
    output_dir_text = tk.Label(master=information_frame, text="")
    output_dir_text.grid(row=1, column=1)
    output_dir_button = tk.Button(master=information_frame, text="Select",
                                  command=lambda: select_directory(output_dir_text))
    output_dir_button.grid(row=1, column=2)

    expert_label = tk.Label(master=information_frame, text="Expert Name: ")
    expert_label.grid(row=2, column=0, sticky='e')
    expert_entry = tk.Entry(master=information_frame)
    expert_entry.grid(row=2, column=1)

    subject_label = tk.Label(master=information_frame, text="Subject Name: ")
    subject_label.grid(row=3, column=0, sticky='e')
    subject_entry = tk.Entry(master=information_frame)
    subject_entry.grid(row=3, column=1)
    information_frame.pack()

    pipeline_label = tk.Label(master=window, text="Pipeline(Default check all)")
    pipeline_label.pack()

    pipeline_frame = tk.Frame(master=window)
    # extract_move_var = tk.IntVar()
    # extract_move_cb = tk.Checkbutton(master=pipeline_frame, text="Extract Move", variable=extract_move_var)
    # extract_move_cb.select()
    # extract_move_cb.grid(row=0, column=0, sticky='w')

    preprocess_var = tk.IntVar()
    preprocess_cb = tk.Checkbutton(master=pipeline_frame, text="Preprocess", variable=preprocess_var)
    preprocess_cb.select()
    preprocess_cb.grid(row=0, column=1, sticky='w')

    predict_var = tk.IntVar()
    predict_cb = tk.Checkbutton(master=pipeline_frame, text="Predict", variable=predict_var)
    predict_cb.select()
    predict_cb.grid(row=0, column=2, stick='w')
    pipeline_frame.pack()

    buttons_frame = tk.Frame()
    buttons_frame.pack(fill=tk.X, ipadx=5, ipady=5)

    btn_submit = tk.Button(master=buttons_frame, text="Run",
                           command=lambda: run(source_dir_text, output_dir_text, expert_entry, subject_entry,
                                               preprocess_var, predict_var, console))

    btn_clear = tk.Button(master=buttons_frame, text="Clear",
                          command=lambda: set_default(source_dir_text, output_dir_text, expert_entry,
                                                      subject_entry, preprocess_cb, predict_cb, console))
    out_cmd_label = tk.Label(master=window, text="Program Output")

    btn_submit.pack(side=tk.RIGHT, padx=10, ipadx=10)
    btn_clear.pack(side=tk.RIGHT, ipadx=10)
    out_cmd_label.pack()
    console = Console(window)
    console.pack(expand=True, fill="both")
    window.mainloop()


if __name__ == '__main__':
    main()
